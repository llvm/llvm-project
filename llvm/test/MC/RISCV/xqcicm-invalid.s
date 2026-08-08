# Xqcicm - Qualcomm uC Conditional Move Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+xqcicm < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-xqcicm < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-MINUS %s

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR from x8 to x15
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.mveqz 9, x10

# CHECK-PLUS: :[[@LINE+2]]:14: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.mveqz x9

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.c.mveqz x9, x10


# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mveq 9, x10, x11, x12

# CHECK-PLUS: :[[@LINE+2]]:11: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mveq x9

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mveq x9, x10, x11, x12


# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvge 9, x10, x11, x12

# CHECK-PLUS: :[[@LINE+2]]:11: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvge x9

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvge x9, x10, x11, x12


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvgeu 9, x10, x11, x12

# CHECK-PLUS: :[[@LINE+2]]:12: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvgeu x9

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvgeu x9, x10, x11, x12


# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvlt 9, x10, x11, x12

# CHECK-PLUS: :[[@LINE+2]]:11: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvlt x9

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvlt x9, x10, x11, x12


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvltu 9, x10, x11, x12

# CHECK-PLUS: :[[@LINE+2]]:12: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvltu x9

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvltu x9, x10, x11, x12


# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvne 9, x10, x11, x12

# CHECK-PLUS: :[[@LINE+2]]:11: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvne x9

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvne x9, x10, x11, x12


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mveqi 9, x10, 5, x12

# CHECK-PLUS: :[[@LINE+2]]:12: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mveqi x9

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-16, 15]
qc.mveqi x9, x10, 17, x12

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mveqi x9, x10, 5, x12


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvgei 9, x10, 5, x12

# CHECK-PLUS: :[[@LINE+2]]:12: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvgei x9

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-16, 15]
qc.mvgei x9, x10, 17, x12

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvgei x9, x10, 5, x12


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvlti 9, x10, 5, x12

# CHECK-PLUS: :[[@LINE+2]]:12: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvlti x9

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-16, 15]
qc.mvlti x9, x10, 17, x12

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvlti x9, x10, 5, x12


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvnei 9, x10, 5, x12

# CHECK-PLUS: :[[@LINE+2]]:12: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvnei x9

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-16, 15]
qc.mvnei x9, x10, 17, x12

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvnei x9, x10, 5, x12


# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvltui 9, x10, 5, x12

# CHECK-PLUS: :[[@LINE+2]]:13: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvltui x9

# CHECK-PLUS: :[[@LINE+1]]:20: error: immediate must be an integer in the range [0, 31]
qc.mvltui x9, x10, 37, x12

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvltui x9, x10, 5, x12


# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvgeui 9, x10, 5, x12

# CHECK-PLUS: :[[@LINE+2]]:13: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.mvgeui x9

# CHECK-PLUS: :[[@LINE+1]]:20: error: immediate must be an integer in the range [0, 31]
qc.mvgeui x9, x10, 37, x12

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvgeui x9, x10, 5, x12
