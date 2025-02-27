# Xqciint - Qualcomm uC Interrupts extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqciint < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqciint < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS %s

# CHECK-PLUS: :[[@LINE+2]]:12: error: immediate must be an integer in the range [0, 1023]
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.setinti 1025

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.setinti 11, 12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.setinti

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.setinti   10


# CHECK-PLUS: :[[@LINE+1]]:12: error: immediate must be an integer in the range [0, 1023]
qc.clrinti 2000

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.clrinti 22, x4

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.clrinti

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.clrinti 8


# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.c.clrint 22

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.clrint

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.clrint x8


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.c.di 22

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.di


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.c.dir 22

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.dir

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.dir x8


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.c.ei 22

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.ei


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.c.eir 22

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.eir

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.eir x8


# CHECK: :[[@LINE+1]]:19: error: invalid operand for instruction
qc.c.mienter.nest 22

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mienter.nest


# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.c.mienter 22

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mienter


# CHECK: :[[@LINE+1]]:17: error: invalid operand for instruction
qc.c.mileaveret 22

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mileaveret


# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.c.setint 22

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.setint

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.setint x8
