# RUN: not llvm-mc %s -triple=riscv64 -mattr=+zmmul -riscv-no-aliases 2>&1 \
# RUN:  | FileCheck -check-prefixes=CHECK-ERROR %s

# CHECK-ERROR: 5:1: error: instruction requires the following: 'M' (Integer Multiplication and Division){{$}}
divw tp, t0, t1

# CHECK-ERROR: 8:1: error: instruction requires the following: 'M' (Integer Multiplication and Division){{$}}
divuw t2, s0, s2

# CHECK-ERROR: 11:1: error: instruction requires the following: 'M' (Integer Multiplication and Division){{$}}
remw a0, a1, a2

# CHECK-ERROR: 14:1: error: instruction requires the following: 'M' (Integer Multiplication and Division){{$}}
remuw a3, a4, a5
