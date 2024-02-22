def compare_files(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            line1 = line1.strip()
            line2 = line2.strip()
            if line1 != line2:
                return False
    return True

# Example usage
file1_path = 'path/to/file1.txt'
file2_path = 'path/to/file2.txt'
are_equal = compare_files(file1_path, file2_path)
print(f"The files are {'equal' if are_equal else 'not equal'}")