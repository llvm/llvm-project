# RUN: echo "// comment" > %t.input
# RUN: which llvm-mc | %python %s %t

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("temp_file")
arguments = parser.parse_args()

llvm_mc_binary = sys.stdin.readlines()[0].strip()
temp_file = arguments.temp_file
input_file = temp_file + ".input"

with open(temp_file, "w") as mc_stdout:
    ## We need to test that starting on an input stream with a non-zero offset
    ## does not trigger an assertion in WinCOFFObjectWriter.cpp, so we seek
    ## past zero for STDOUT.
    mc_stdout.seek(4)
    subprocess.run(
        [llvm_mc_binary, "-filetype=obj", "-triple", "i686-pc-win32", input_file],
        stdout=mc_stdout,
        check=True,
    )
