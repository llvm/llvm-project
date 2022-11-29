import os
import subprocess
import argparse
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description='Input Fuzzer')
    parser.add_argument('-f', '--file',
                        help='Input C/C++ source file absolute path.')
    parser.add_argument('-j', '--count',
                        help='Number of input variables',
                        type=int,
                        default=1)
    parser.add_argument('-c', '--inputs',
                        help='Input intervals for each input.',
                        type=float,
                        nargs='*',
                        default=1)
    parser.add_argument('-i', '--interval',
                        help='Interval between input values',
                        type=float,
                        nargs='*',
                        default=1)
    parser.add_argument('-m', '--midpoint',
                        help='Midpoint value of the range around which to sample input points.',
                        type=float,
                        nargs='*',
                        default=1)
    parser.add_argument('-n', '--numpoints',
                        help='Number of points on one side of the midpoint.',
                        type=int,
                        nargs='*',
                        default=1)
    parser.add_argument('-p', '--index',
                        help='The index of operand in the parameter list of the function.',
                        type=int,
                        default=0)
    arguments = parser.parse_args()
    file_path = Path(arguments.file)
    interval = arguments.interval
    mid_point = arguments.midpoint
    num_points_one_side = arguments.numpoints
    num_variables = arguments.count

    # Changing Working Directory to where the source file is present
    os.chdir(file_path.parent)

    # Building the program
    subprocess.run(['make', 'build_pass', 'NO_DATA_DUMP=-DNO_DATA_DUMP', 'MEMORY_OPT=-DMEMORY_OPT'])

    print(file_path.parent)

    # Generating Inputs
    inputs = [[mid_point[j] + i * interval[j] for i in range(-num_points_one_side[j], num_points_one_side[j] + 1)]
              for j in range(num_variables)]
    print(inputs)
    
    # Running the program
    for i in range(len(inputs[0])):
        input_string = ''
        for variable_value_list in inputs:
            input_string += str(variable_value_list[i]) + '\n'
        proc = subprocess.run(['make', 'run_pass2'], input=bytes(input_string, 'utf-8'))


if __name__ == "__main__":
    main()
