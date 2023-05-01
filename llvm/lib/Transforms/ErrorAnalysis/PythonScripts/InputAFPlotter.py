import json
import argparse
import matplotlib.pyplot as plt


def load_data(file_name, call_number, operand_number):
    af_data = json.load(open(file_name, 'r'))

    # Filtering a particular call out of all calls of the function using its index in the Call chain (Made up thing)
    call_dictionary = dict(filter(lambda elem: elem[0].split('_')[2] == call_number, af_data.items()))

    return [program_record[operand_number]['AF'] for program_record in call_dictionary.values()]


def main():
    parser = argparse.ArgumentParser(description='Input Fuzzer')
    parser.add_argument('-f', '--file',
                        help='Input JSON file containing AF data')
    parser.add_argument('-i', '--interval',
                        help='Interval between input values',
                        type=float,
                        default=1)
    parser.add_argument('-m', '--midpoint',
                        help='Midpoint value of the range around which to sample input points.',
                        type=float,
                        default=1)
    parser.add_argument('-n', '--numpoints',
                        help='Number of points on one side of the midpoint.',
                        type=int,
                        default=1)
    parser.add_argument('-p', '--index',
                        help='The index of operand in the parameter list of the function.',
                        type=int,
                        default=0)
    arguments = parser.parse_args()
    
    interval = arguments.interval
    mid_point = arguments.midpoint
    num_points_one_side = arguments.numpoints

    x_axis = [mid_point + i*interval for i in range(-num_points_one_side, num_points_one_side + 1)]
    print(x_axis)
    y_axis = load_data(arguments.file,
                       "0", arguments.index)
    
    fig = plt.figure()
    plt.scatter(x_axis, y_axis, color='maroon', marker='o')
    plt.xlabel('Variable')
    plt.ylabel('AF')
    plt.show()


if __name__ == "__main__":
    main()
