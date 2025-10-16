#!/usr/bin/env python3

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_args",
        type=int,
        help="Max number of arguments to generate case statements for",
        required=True,
    )
    parser.add_argument("--output", help="Output header file to include", required=True)
    args = parser.parse_args()

    output = ""
    for i in range(args.max_args + 1):
        output += "case %d:\n" % (i)
        output += "((void (*)(kmp_int32 *, kmp_int32 *\n"
        for j in range(i):
            output += ", void *"
            if (j + 1) % 4 == 0:
                output += "\n"
        output += "))fn)(&global_tid, &bound_tid\n"
        for j in range(i):
            output += ", args[%d]" % (j)
            if (j + 1) % 4 == 0:
                output += "\n"
        output += ");\n"
        output += "break;\n"

    with open(args.output, "w") as f:
        print(output, file=f)


if __name__ == "__main__":
    main()
