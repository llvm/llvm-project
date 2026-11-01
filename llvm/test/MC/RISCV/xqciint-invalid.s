# Xqciint - Qualcomm uC Interrupts extension
# RUN: not llvm-mc -triple riscv32 -mattr=+xqciint < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-xqciint < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-MINUS %s

# CHECK-PLUS: :[[@LINE+2]]:12: error: immediate must be an integer in the range [0, 1023]
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.setinti 1025

# CHECK-PLUS: :[[@LINE+2]]:16: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.setinti 11, 12

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.setinti

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.setinti   10


# CHECK-PLUS: :[[@LINE+1]]:12: error: immediate must be an integer in the range [0, 1023]
qc.clrinti 2000

# CHECK-PLUS: :[[@LINE+2]]:16: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.clrinti 22, x4

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.clrinti

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.clrinti 8


# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.clrint 22

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.clrint

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.clrint x8


# CHECK-PLUS: :[[@LINE+2]]:9: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.di 22

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.di


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.dir 22

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.dir

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.dir x8


# CHECK-PLUS: :[[@LINE+2]]:9: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.ei 22

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.ei


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.eir 22

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.eir

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.eir x8


# CHECK-PLUS: :[[@LINE+2]]:19: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.mienter.nest 22

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mienter.nest


# CHECK-PLUS: :[[@LINE+2]]:14: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.mienter 22

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mienter


# CHECK-PLUS: :[[@LINE+2]]:17: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.mileaveret 22

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mileaveret


# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.setint 22

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.setint

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.setint x8


# CHECK-PLUS: :[[@LINE+2]]:11: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.mret x8

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mret


# CHECK-PLUS: :[[@LINE+2]]:12: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.mnret 10

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mnret
