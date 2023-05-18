"""
Script to disassembles a bitcode file and run FileCheck on the output with the
provided arguments. The first 2 arguments are the paths to the llvm-dis and
FileCheck binaries, followed by arguments to be passed to FileCheck. The last
argument is the bitcode file to disassemble.

Usage:
    python llvm-dis-and-filecheck.py
      <path to llvm-dis> <path to FileCheck>
      [arguments passed to FileCheck] <path to bitcode file>

"""


import sys
import os
import subprocess

llvm_dis = sys.argv[1]
filecheck = sys.argv[2]
filecheck_args = [
    filecheck,
]
filecheck_args.extend(sys.argv[3:-1])
bitcode_file = sys.argv[-1]
ir_file = bitcode_file + ".ll"

disassemble = subprocess.Popen([llvm_dis, "--preserve-ll-uselistorder", "-o", ir_file, bitcode_file])
if os.path.exists(ir_file + ".0"):
    ir_file = ir_file + ".0"

disassemble.communicate()

if disassemble.returncode != 0:
    print("stderr:")
    print(disassemble.stderr)
    print("stdout:")
    print(disassemble.stdout)
    sys.exit(1)

check = None
with open(ir_file, "r") as ir:
    check = subprocess.Popen(filecheck_args, stdin=ir, stdout=sys.stdout)
check.communicate()
sys.exit(check.returncode)
