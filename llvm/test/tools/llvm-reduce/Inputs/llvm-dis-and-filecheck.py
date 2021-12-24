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
import subprocess

llvm_dis = sys.argv[1]
filecheck = sys.argv[2]
filecheck_args = [filecheck, ]
filecheck_args.extend(sys.argv[3:-1])
bitcode_file = sys.argv[-1]

disassemble = subprocess.Popen([llvm_dis, "-o", "-", bitcode_file],
        stdout=subprocess.PIPE)
check = subprocess.Popen(filecheck_args, stdin=disassemble.stdout)
disassemble.stdout.close()
check.communicate()
sys.exit(check.returncode)
