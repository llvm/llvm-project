# Xqcicm - Qualcomm uC Conditional Move Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcicm < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-IMM %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcicm < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-EXT %s

# CHECK: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.c.mveqz 9, x10

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.mveqz x9

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.c.mveqz x9, x10


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.mveq 9, x10, x11, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mveq x9

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mveq x9, x10, x11, x12


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.mvge 9, x10, x11, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mvge x9

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvge x9, x10, x11, x12


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.mvgeu 9, x10, x11, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mvgeu x9

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvgeu x9, x10, x11, x12


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.mvlt 9, x10, x11, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mvlt x9

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvlt x9, x10, x11, x12


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.mvltu 9, x10, x11, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mvltu x9

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvltu x9, x10, x11, x12


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.mvne 9, x10, x11, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mvne x9

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvne x9, x10, x11, x12


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.mveqi 9, x10, 5, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mveqi x9

# CHECK-IMM: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-16, 15]
qc.mveqi x9, x10, 17, x12

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mveqi x9, x10, 5, x12


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.mvgei 9, x10, 5, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mvgei x9

# CHECK-IMM: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-16, 15]
qc.mvgei x9, x10, 17, x12

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvgei x9, x10, 5, x12


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.mvlti 9, x10, 5, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mvlti x9

# CHECK-IMM: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-16, 15]
qc.mvlti x9, x10, 17, x12

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvlti x9, x10, 5, x12


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.mvnei 9, x10, 5, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mvnei x9

# CHECK-IMM: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-16, 15]
qc.mvnei x9, x10, 17, x12

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvnei x9, x10, 5, x12


# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.mvltui 9, x10, 5, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mvltui x9

# CHECK-IMM: :[[@LINE+1]]:20: error: immediate must be an integer in the range [0, 31]
qc.mvltui x9, x10, 37, x12

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvltui x9, x10, 5, x12


# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.mvgeui 9, x10, 5, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.mvgeui x9

# CHECK-IMM: :[[@LINE+1]]:20: error: immediate must be an integer in the range [0, 31]
qc.mvgeui x9, x10, 37, x12

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)
qc.mvgeui x9, x10, 5, x12
