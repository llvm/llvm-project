# RUN: echo "// comment" > %t.input
# We use STDIN to for the binary name as lit will substitute in the full path
# of the binary before executing, ensuring we pick up the correct llvm-mc.
# RUN: echo llvm-mc | %python %s %t.input %t

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("temp_file")
arguments = parser.parse_args()

llvm_mc_binary = sys.stdin.readlines()[0].strip()

with open(arguments.temp_file, "w") as mc_stdout:
    ## We need to test that starting on an input stream with a non-zero offset
    ## does not trigger an assertion in WinCOFFObjectWriter.cpp, so we seek
    ## past zero for STDOUT.
    mc_stdout.seek(4)
    subprocess.run(
        [
            llvm_mc_binary,
            "-filetype=obj",
            "-triple",
            "i686-pc-win32",
            arguments.input_file,
        ],
        stdout=mc_stdout,
        check=True,
    )
