# Xqcicli - Qualcomm uC Conditional Load Immediate Instructions
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcicli < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcicli < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS %s

# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.lieq x0, x4, x6, 10

# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.lieq x2, x0, x6, 10

# CHECK: :[[@LINE+1]]:17: error: invalid operand for instruction
qc.lieq x2, x4, x0, 10

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lieq x2, x4, x6

# CHECK-PLUS: :[[@LINE+1]]:21: error: immediate must be an integer in the range [-16, 15]
qc.lieq x2, x4, x6, 40

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.lieq x2, x4, x6, 10


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.lige x0, x8, x20, 2

# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.lige x4, x0, x20, 2

# CHECK: :[[@LINE+1]]:17: error: invalid operand for instruction
qc.lige x4, x8, x0, 2

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lige x4, x8, x20

# CHECK-PLUS: :[[@LINE+1]]:22: error: immediate must be an integer in the range [-16, 15]
qc.lige x4, x8, x20, -18

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.lige x4, x8, x20, 2


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.lilt x0, x9, x10, 3

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.lilt x19, x0, x10, 3

# CHECK: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.lilt x19, x9, x0, 3

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lilt x19, x9, x10

# CHECK-PLUS: :[[@LINE+1]]:23: error: immediate must be an integer in the range [-16, 15]
qc.lilt x19, x9, x10, 39

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.lilt x19, x9, x10, 3


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.line x0, x14, x6, 10

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.line x18, x0, x6, 10

# CHECK: :[[@LINE+1]]:19: error: invalid operand for instruction
qc.line x18, x14, x0, 10

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.line x18, x14, x6

# CHECK-PLUS: :[[@LINE+1]]:23: error: immediate must be an integer in the range [-16, 15]
qc.line x18, x14, x6, 100

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.line x18, x14, x6, 10


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.ligeu x0, x4, x6, 10

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.ligeu x2, x0, x6, 10

# CHECK: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.ligeu x2, x4, x0, 10

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.ligeu x2, x4, x6

# CHECK-PLUS: :[[@LINE+1]]:22: error: immediate must be an integer in the range [-16, 15]
qc.ligeu x2, x4, x6, 70

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.ligeu x2, x4, x6, 10


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.liltu x0, x19, x12, 13

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.liltu x1, x0, x12, 13

# CHECK: :[[@LINE+1]]:19: error: invalid operand for instruction
qc.liltu x1, x19, x0, 13

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.liltu x1, x19, x12

# CHECK-PLUS: :[[@LINE+1]]:24: error: immediate must be an integer in the range [-16, 15]
qc.liltu x1, x19, x12, 73

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.liltu x1, x19, x12, 13


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.lieqi x0, x1, 15, 12

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.lieqi x7, x0, 15, 12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lieqi x7, x1, 15

# CHECK-PLUS: :[[@LINE+1]]:18: error: immediate must be an integer in the range [-16, 15]
qc.lieqi x7, x1, 25, 12

# CHECK-PLUS: :[[@LINE+1]]:22: error: immediate must be an integer in the range [-16, 15]
qc.lieqi x7, x1, 15, -22

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.lieqi x7, x1, 15, 12


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.ligei x0, x11, -4, 9

# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.ligei x17, x0, -4, 9

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.ligei x17, x11, -4

# CHECK-PLUS: :[[@LINE+1]]:20: error: immediate must be an integer in the range [-16, 15]
qc.ligei x17, x11, -24, 9

# CHECK-PLUS: :[[@LINE+1]]:24: error: immediate must be an integer in the range [-16, 15]
qc.ligei x17, x11, -4, 59

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.ligei x17, x11, -4, 9


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.lilti x0, x11, -14, 2

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.lilti x9, x0, -14, 2

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lilti x9, x11, -14

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-16, 15]
qc.lilti x9, x11, -84, 2

# CHECK-PLUS: :[[@LINE+1]]:24: error: immediate must be an integer in the range [-16, 15]
qc.lilti x9, x11, -14, 52

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.lilti x9, x11, -14, 2


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.linei x0, x1, 10, 12

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.linei x5, x0, 10, 12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.linei x5, x1, 10

# CHECK-PLUS: :[[@LINE+1]]:18: error: immediate must be an integer in the range [-16, 15]
qc.linei x5, x1, 130, 12

# CHECK-PLUS: :[[@LINE+1]]:22: error: immediate must be an integer in the range [-16, 15]
qc.linei x5, x1, 10, 124

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.linei x5, x1, 10, 12


# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.ligeui x0, x12, 7, -12

# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.ligeui x2, x0, 7, -12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.ligeui x2, x12, 7

# CHECK-PLUS: :[[@LINE+1]]:20: error: immediate must be an integer in the range [0, 31]
qc.ligeui x2, x12, -7, -12

# CHECK-PLUS: :[[@LINE+1]]:23: error: immediate must be an integer in the range [-16, 15]
qc.ligeui x2, x12, 7, -17

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.ligeui x2, x12, 7, -12


# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.liltui x0, x25, 31, 12

# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.liltui x3, x0, 31, 12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.liltui x3, x25, 31

# CHECK-PLUS: :[[@LINE+1]]:20: error: immediate must be an integer in the range [0, 31]
qc.liltui x3, x25, 32, 12

# CHECK-PLUS: :[[@LINE+1]]:24: error: immediate must be an integer in the range [-16, 15]
qc.liltui x3, x25, 31, 112

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)
qc.liltui x3, x25, 31, 12
