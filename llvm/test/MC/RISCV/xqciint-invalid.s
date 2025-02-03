# Xqciint - Qualcomm uC Interrupts extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqciint < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-IMM %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqciint < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-EXT %s

# CHECK-IMM: :[[@LINE+1]]:12: error: immediate must be an integer in the range [0, 1023]
qc.setinti 1025

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.setinti 11, 12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.setinti

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.setinti   10


# CHECK-IMM: :[[@LINE+1]]:12: error: immediate must be an integer in the range [0, 1023]
qc.clrinti 2000

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.clrinti 22, x4

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.clrinti

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.clrinti 8


# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.c.clrint 22

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.clrint

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.clrint x8


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.c.di 22

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.di


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.c.dir 22

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.dir

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.dir x8


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.c.ei 22

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.ei


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.c.eir 22

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.eir

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.eir x8


# CHECK: :[[@LINE+1]]:19: error: invalid operand for instruction
qc.c.mienter.nest 22

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mienter.nest


# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.c.mienter 22

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mienter


# CHECK: :[[@LINE+1]]:17: error: invalid operand for instruction
qc.c.mileaveret 22

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mileaveret


# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.c.setint 22

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.setint

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.setint x8
