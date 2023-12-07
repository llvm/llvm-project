"""
Script to assemble a text IR file and run FileCheck on the output with the
provided arguments. The first 2 arguments are the paths to the llvm-as and
FileCheck binaries, followed by arguments to be passed to FileCheck. The last
argument is the text IR file to disassemble.

Usage:
    python llvm-as-and-filecheck.py
      <path to llvm-as> <path to FileCheck>
      [arguments passed to FileCheck] <path to text IR file>

"""
import sys
import os
import subprocess

llvm_as = sys.argv[1]
filecheck = sys.argv[2]
filecheck_args = [
    filecheck
]

filecheck_args.extend(sys.argv[3:-1])
ir_file = sys.argv[-1]
bitcode_file = ir_file + ".bc"

# Verify the IR actually parses since FileCheck is too dumb to know.
assemble = subprocess.Popen([llvm_as, "-o", bitcode_file, ir_file])
assemble.communicate()

if assemble.returncode != 0:
    print("stderr:")
    print(assemble.stderr)
    print("stdout:")
    print(assemble.stdout)
    sys.exit(0)

filecheck_args.append("--input-file")
filecheck_args.append(ir_file)

check = subprocess.Popen(filecheck_args)
check.communicate()
sys.exit(check.returncode)
