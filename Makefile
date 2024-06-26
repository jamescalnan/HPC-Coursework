# Makefile for 2D Advection Example Program

# Compiler settings
CC = gcc
CFLAGS = -std=c99 -Wall -fopenmp
LDFLAGS = -lm -fopenmp

# Target executable name
TARGET = advection2D

# Source and object files
SRC = advection2D.c
OBJ = $(SRC:.c=.o)

# Default target
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# To remove compiled files
clean:
	rm -f $(TARGET) $(OBJ) initial.dat final.dat averaged.dat

# Dependencies
$(OBJ): $(SRC)

# Phony targets
.PHONY: all clean
