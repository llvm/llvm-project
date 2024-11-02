# RUN: not llvm-mc %s -triple=riscv32 -mattr=+zmmul -riscv-no-aliases 2>&1 \
# RUN:  | FileCheck -check-prefixes=CHECK-ERROR %s

# CHECK-ERROR: 5:1: error: instruction requires the following: 'M' (Integer Multiplication and Division){{$}}
div s0, s0, s0

# CHECK-ERROR: 8:1: error: instruction requires the following: 'M' (Integer Multiplication and Division){{$}}
divu gp, a0, a1

# CHECK-ERROR: 11:1: error: instruction requires the following: 'M' (Integer Multiplication and Division){{$}}
rem s2, s2, s8

# CHECK-ERROR: 14:1: error: instruction requires the following: 'M' (Integer Multiplication and Division){{$}}
remu x18, x18, x24
