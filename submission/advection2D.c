/*******************************************************************************
2D advection example program which advects a Gaussian u(x,y) at a fixed velocity



Outputs: initial.dat - inital values of u(x,y) 
         final.dat   - final values of u(x,y)

         The output files have three columns: x, y, u

         Compile with: gcc -o advection2D -std=c99 advection2D.c -lm

Notes: The time step is calculated using the CFL condition

********************************************************************************/

/*********************************************************************
                     Include header files 
**********************************************************************/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <time.h>

/*********************************************************************
                     Define constants
**********************************************************************/

#define BUFFER_SIZE 1024

/*********************************************************************
                     Function prototypes
**********************************************************************/

int compareFiles(const char* currentFilePath, const char* referenceFilePath);

/*********************************************************************
                      Main function
**********************************************************************/

int main(){

  /* Modified Grid properties to cover the range 0 ≤ x ≤ 30.0 m and 0 ≤ y ≤ 30.0 m */
  const int NX=1000;    // Number of x points
  const int NY=1000;    // Number of y points
  const float xmin=0.0; // Minimum x value
  const float xmax=30.0; // Maximum x value, changed from 1.0 to 30.0
  const float ymin=0.0; // Minimum y value
  const float ymax=30.0; // Maximum y value, changed from 1.0 to 30.0
  
  /* Modified Parameters for the Gaussian initial conditions to represent a cloud */
  const float x0=3.0;                    // Centre(x), changed to 3.0 m
  const float y0=15.0;                   // Centre(y), changed to 15.0 m (vertically centered)
  const float sigmax=1.0;                // Width(x), changed to 1.0 m
  const float sigmay=5.0;                // Width(y), changed to 5.0 m (larger vertical extent)
  const float sigmax2 = sigmax * sigmax; // Width(x) squared
  const float sigmay2 = sigmay * sigmay; // Width(y) squared


  /* Boundary conditions */
  const float bval_left=0.0;    // Left boudnary value
  const float bval_right=0.0;   // Right boundary value
  const float bval_lower=0.0;   // Lower boundary
  const float bval_upper=0.0;   // Upper bounary
  
  /* Time stepping parameters */
  const float CFL=0.9;   // CFL number 
  const int nsteps=800; // Number of time steps

  /* Modified velocities for horizontal advection and no vertical movement */
  const float velx=1.0; // Velocity in x direction, changed to 1.0 m/s
  const float vely=0.0; // Velocity in y direction, changed to 0. (no vertical movement)

  const double z_0 = 1.0;     // Roughness length
  const double u_star = 0.2; // Friction velocity
  const double kappa = 0.41;  // Von Karman constant

  /* Arrays to store variables. These have NX+2 elements
     to allow boundary values to be stored at both ends */
  float x[NX+2];          // x-axis values
  float y[NX+2];          // y-axis values
  float u[NX+2][NY+2];    // Array of u values
  float dudt[NX+2][NY+2]; // Rate of change of u
  float u_avg[NX+2];      // Array to store the vertically averaged values of u


  // float x2;   // x squared (used to calculate iniital conditions)
  // float y2;   // y squared (used to calculate iniital conditions)
  
  /* Calculate distance between points */
  float dx = (xmax-xmin) / ( (float) NX);
  float dy = (ymax-ymin) / ( (float) NY);
  
  /* Calculate time step using the CFL condition */
  /* The fabs function gives the absolute value in case the velocity is -ve */
  float dt = CFL / ( (fabs(velx) / dx) + (fabs(vely) / dy) );
  
  /*** Report information about the calculation ***/
  printf("Grid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("Time step           = %g\n", dt);
  printf("No. of time steps   = %d\n", nsteps);
  printf("End time            = %g\n", dt*(float) nsteps);
  printf("Distance advected x = %g\n", velx*dt*(float) nsteps);
  printf("Distance advected y = %g\n", vely*dt*(float) nsteps);

  /*** Place x points in the middle of the cell ***/
  /* LOOP 1 */
  // Since there are no dependencies between the iterations of this loop, it can be parallelised
  #pragma omp parallel for
  for (int i=0; i<NX+2; i++){
    x[i] = ( (float) i - 0.5) * dx;
  }

  /*** Place y points in the middle of the cell ***/
  /* LOOP 2 */
  // Since there are no dependencies between the iterations of this loop, it can be parallelised
  #pragma omp parallel for
  for (int j=0; j<NY+2; j++){
    y[j] = ( (float) j - 0.5) * dy;
  }

  /*** Set up Gaussian initial conditions ***/
  /* LOOP 3 */
  // Each computation of u[i][j] is independent of the others, so this loop can be parallelised
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < NX + 2; i++) {
      for (int j = 0; j < NY + 2; j++) {
          float x2 = (x[i] - x0) * (x[i] - x0);
          float y2 = (y[j] - y0) * (y[j] - y0);
          u[i][j] = exp(-1.0 * ((x2 / (2.0 * sigmax2)) + (y2 / (2.0 * sigmay2))));
      }
  }

  /*** Write array of initial u values out to file ***/
  FILE *initialfile;
  initialfile = fopen("initial.dat", "w");
  /* LOOP 4 */
  // Because of how file streams work, this loop cannot be parallelised as it would jumble the output order
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(initialfile);


  // Compare the initial file with the reference file
  // const char* currentFile = "initial.dat";
  // const char* referenceFile = "/home/links/jdc235/HPC-Coursework/demo_answers/initial.dat";

  // if(compareFiles(currentFile, referenceFile) == 0) {
  //     printf("Initial files are identical.\n");
  // } else {
  //     printf("Initial files are different.\n");
  // }
  // End of comparison


  // Store the max velocity from the result on eq 1 to be used to calculate the time step
  float max_velx = 0.0;


  /* LOOP 5 */
  // This isn't parallelisable becasue each iteration can depend on the previous one and parallelising it could cause a race condition
  for (int m=0; m<nsteps; m++){
    /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
    /* LOOP 6 */
    // This loop can be parallelised as each iteration is independent of the others
    #pragma omp parallel for
    for (int j=0; j<NY+2; j++){
      u[0][j]    = bval_left;
      u[NX+1][j] = bval_right;
    }

    /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
    /* LOOP 7 */
    // This loop can be parallelised as each iteration is independent of the others
    #pragma omp parallel for
    for (int i=0; i<NX+2; i++){
      u[i][0]    = bval_lower;
      u[i][NY+1] = bval_upper;
    }
    
    /*** Calculate rate of change of u using leftward difference ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 8 */
    // This loop can be parallelised as each iteration is independent of the others
    // The calculation for each cell only depends on its immediate neighbors, 
    // which are not being modified in this loop. Thus, iterations can be executed in 
    // parallel without causing data races.
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NX + 1; i++) {
      for (int j = 1; j < NY + 1; j++) {
        float z = y[j];
        // ln(0) = error 
        float horizontal_velocity;
        if (z > z_0) {
          // Chekc if z / z_0 is 0 and if so set horizontal_velocity to 0
          horizontal_velocity = (u_star / kappa) * log(z / z_0); // eq 1
          
        } else if (z <= z_0) {
          horizontal_velocity = 0.0;
        }

        dudt[i][j] =  - horizontal_velocity * (u[i][j] - u[i - 1][j]) / dx
                      - vely * (u[i][j] - u[i][j - 1]) / dy;
        max_velx = fmax(max_velx, fabs(horizontal_velocity));
      }
    }


    // Recalculate the time step using the CFL condition with the correct maximum velocity
    dt = CFL / ((fabs(max_velx) / dx) + (fabs(vely) / dy));

    /*** Update u from t to t+dt ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 9 */
    // This loop can be parallelised as the update of u[i][j] is based on already calculated 
    // values of dudt[i][j] and does not modify the dudt array itself.
    #pragma omp parallel for collapse(2)
    for	(int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
        u[i][j] = u[i][j] + dudt[i][j] * dt;
      }
    }

    // Calculate the vertically averaged distribution, task 2.4 (doesnt need to be parallelised)
    for (int i = 1; i <= NX; i++) {
        float sum = 0.0;
        for (int j = 1; j <= NY; j++) {
            sum += u[i][j];
        }
        u_avg[i] = sum / NY; // Averaging over all y values
    }
    
  } // time loop
  

  // Output the vertically averaged distribution to a file
  FILE *avgfile;
  avgfile = fopen("averaged.dat", "w");
  for (int i = 1; i <= NX; i++) {
      fprintf(avgfile, "%g %g\n", x[i], u_avg[i]);
  }
  fclose(avgfile);

  /*** Write array of final u values out to file ***/
  FILE *finalfile;
  finalfile = fopen("final.dat", "w");
  /* LOOP 10 */
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){

      fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(finalfile);

  // Compare the final file with the reference file
  // currentFile = "final.dat";
  // referenceFile = "/home/links/jdc235/HPC-Coursework/demo_answers/final.dat";

  // if(compareFiles(currentFile, referenceFile) == 0) {
  //     printf("Final files are identical.\n");
  // } else {
  //     printf("Final files are different.\n");
  // }
  // End of comparison


  return 0;
}


/*********************************************************************
                     Function definitions
**********************************************************************/

// Function to compare two files
int compareFiles(const char* currentFilePath, const char* referenceFilePath) {
    FILE *file1, *file2;
    char buffer1[BUFFER_SIZE], buffer2[BUFFER_SIZE];

    // Open the current file
    file1 = fopen(currentFilePath, "r");
    if (!file1) {
        perror("Failed to open current file");
        return -1; // Return -1 on error
    }

    // Open the reference file
    file2 = fopen(referenceFilePath, "r");
    if (!file2) {
        perror("Failed to open reference file");
        fclose(file1); // Make sure to close the first file before returning
        return -1; // Return -1 on error
    }

    // Compare the files line by line
    while (fgets(buffer1, BUFFER_SIZE, file1) && fgets(buffer2, BUFFER_SIZE, file2)) {
        if (strcmp(buffer1, buffer2) != 0) {
            // If any line doesn't match, close both files and return 1
            fclose(file1);
            fclose(file2);

            // Output the line that didn't match
            printf("Line in current file: %s", buffer1);
            printf("Line in reference file: %s", buffer2);


            return 1; // Files are different
        }
    }

    // Close the files
    fclose(file1);
    fclose(file2);

    return 0; // Files are identical
}


/* End of file ******************************************************/