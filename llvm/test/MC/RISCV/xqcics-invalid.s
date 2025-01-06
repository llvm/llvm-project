# Xqcics - Qualcomm uC Conditional Select Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcics < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-IMM %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcics < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-EXT %s

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.selecteqi 9, 15, x4, x3

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.selecteqi x9, 15, x4

# CHECK-IMM: :[[@LINE+1]]:18: error: immediate must be an integer in the range [-16, 15]
qc.selecteqi x9, 16, x4, x3

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)
qc.selecteqi x9, 15, x4, x3


# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.selectieq 8, x4, x3, 12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.selectieq x8, x4, x3

# CHECK-IMM: :[[@LINE+1]]:26: error: immediate must be an integer in the range [-16, 15]
qc.selectieq x8, x4, x3, 17

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)
qc.selectieq x8, x4, x3, 12


# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.selectieqi 9, 11, x3, 12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.selectieqi x9, 11, x3

# CHECK-IMM: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-16, 15]
qc.selectieqi x9, 16, x3, 12

# CHECK-IMM: :[[@LINE+1]]:27: error: immediate must be an integer in the range [-16, 15]
qc.selectieqi x9, 11, x3, 18

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)
qc.selectieqi x9, 11, x3, 12


# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.selectiieq 9, x3, 11, 12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.selectiieq x9, x3, 11

# CHECK-IMM: :[[@LINE+1]]:23: error: immediate must be an integer in the range [-16, 15]
qc.selectiieq x9, x3, 16, 12

# CHECK-IMM: :[[@LINE+1]]:27: error: immediate must be an integer in the range [-16, 15]
qc.selectiieq x9, x3, 11, 17

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)
qc.selectiieq x9, x3, 11, 12


# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.selectiine 8, x3, 10, 11

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.selectiine x8, x3, 10

# CHECK-IMM: :[[@LINE+1]]:23: error: immediate must be an integer in the range [-16, 15]
qc.selectiine x8, x3, 16, 11

# CHECK-IMM: :[[@LINE+1]]:27: error: immediate must be an integer in the range [-16, 15]
qc.selectiine x8, x3, 12, 18

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)
qc.selectiine x8, x3, 10, 11


# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.selectine 8, x3, x4, 11

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.selectine x8, x3, x4

# CHECK-IMM: :[[@LINE+1]]:26: error: immediate must be an integer in the range [-16, 15]
qc.selectine x8, x3, x4, 16

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)
qc.selectine x8, x3, x4, 11


# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.selectinei 8, 11, x3, 12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.selectinei x8, 11, x3

# CHECK-IMM: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-16, 15]
qc.selectinei x8, 16, x3, 12

# CHECK-IMM: :[[@LINE+1]]:27: error: immediate must be an integer in the range [-16, 15]
qc.selectinei x8, 11, x3, 18

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)
qc.selectinei x8, 11, x3, 12


# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.selectnei 8, 11, x3, x5

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.selectnei x8, 11, x3

# CHECK-IMM: :[[@LINE+1]]:18: error: immediate must be an integer in the range [-16, 15]
qc.selectnei x8, 16, x3, x5

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)
qc.selectnei x8, 11, x3, x5

