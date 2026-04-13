#!/usr/bin/env python3

from pathlib import Path
from os import path
from argparse import ArgumentParser

desc = """
A simple script to create new llvm-mca tests from a set of input files. For each .s files under <input files folder>,
it creates a corresponding .test file containing the RUN line, in which the llvm-mca command reads the input from the
said .s file.

Example Usage:
```
create_template_mca_tests.py --triple riscv64 --cpu generic-rv64 test/tools/llvm-mca/RISCV/Inputs test/tools/llvm-mca/RISCV/Generic
```

This command will create files like `test/tools/llvm-mca/RISCV/Generic/mul-div.test` whose RUN line takes
`test/tools/llvm-mca/RISCV/Inputs/mul-div.s` as input.
"""

run_line_template = "# RUN: llvm-mca -mtriple={triple} -mcpu={cpu} -iterations=1 -instruction-tables=full {extra_mca_args} %p/{input_file} | FileCheck {extra_filecheck_args} %s"


def entry(args):
    input_dir_path = Path(args.input_dir)
    output_dir_path = Path(args.output_dir)

    # Find all the input files.
    for input_file in input_dir_path.glob("**/*.s"):
        new_file = input_file.relative_to(input_dir_path)
        # Filter out unwanted files.
        if any(new_file.match(x) for x in (args.exclude or [])):
            continue
        new_file = output_dir_path / new_file.with_suffix(".test")
        # Make sure the destination folder is there
        new_file.parent.mkdir(exist_ok=True, parents=True)
        with open(new_file, "w") as output_file:
            input_relative_path = path.relpath(input_file, start=new_file.parent)
            run_line = run_line_template.format(
                triple=args.triple,
                cpu=args.cpu,
                input_file=input_relative_path,
                extra_mca_args=args.extra_mca_args,
                extra_filecheck_args=args.extra_filecheck_args,
            )
            print(run_line, file=output_file)


if __name__ == "__main__":
    parser = ArgumentParser(description=desc)
    parser.add_argument(
        "input_dir",
        type=str,
        metavar="<input files folder>",
        help="Folder to read input files from",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        metavar="<output folder>",
        help="Folder to put output test files to",
    )
    parser.add_argument("--triple", required=True, type=str, help="-mtriple to use")
    parser.add_argument("--cpu", required=True, type=str, help="-mcpu to use")
    parser.add_argument(
        "--exclude",
        nargs="+",
        metavar="<file>",
        help="Input files, relative to the input folder, to exclude",
    )
    parser.add_argument(
        "--extra-mca-args",
        type=str,
        default="",
        metavar="<flags>",
        help="Extra command line arguments for llvm-mca",
    )
    parser.add_argument(
        "--extra-filecheck-args",
        type=str,
        default="",
        metavar="<flags>",
        help="Extra command line arguments for FileCheck",
    )
    entry(parser.parse_args())
