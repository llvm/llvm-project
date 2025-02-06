# Xqcilsm - Qualcomm uC Load Store Multiple Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcilsm < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcilsm < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS %s

# CHECK: :[[@LINE+1]]:20: error: expected register
qc.swm x5, x20, 12(20)

# CHECK: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.swm x5, x0, 12(x3)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.swm x5, x3

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be a multiple of 4 bytes in the range [0, 124]
qc.swm x5, x20, 45(x3)

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)
qc.swm x5, x20, 12(x3)


# CHECK: :[[@LINE+1]]:20: error: expected register
qc.swmi x10, 4, 20(4)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.swmi x10, 4, 20

# CHECK-PLUS: :[[@LINE+1]]:14: error: immediate must be an integer in the range [1, 31]
qc.swmi x10, 32, 20(x4)

# CHECK-PLUS: :[[@LINE+1]]:14: error: immediate must be an integer in the range [1, 31]
qc.swmi x10, 0, 20(x4)

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be a multiple of 4 bytes in the range [0, 124]
qc.swmi x10, 4, 45(x4)

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)
qc.swmi x10, 4, 20(x4)


# CHECK: :[[@LINE+1]]:23: error: expected register
qc.setwm x4, x30, 124(2)

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.setwm x4, x0, 124(x2)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.setwm x4, x30, 124

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be a multiple of 4 bytes in the range [0, 124]
qc.setwm x4, x30, 128(x2)

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)
qc.setwm x4, x30, 124(x2)


# CHECK: :[[@LINE+1]]:22: error: expected register
qc.setwmi x5, 31, 12(12)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.setwmi x5, 31, 12

# CHECK-PLUS: :[[@LINE+1]]:15: error: immediate must be an integer in the range [1, 31]
qc.setwmi x5, 37, 12(x12)

# CHECK-PLUS: :[[@LINE+1]]:15: error: immediate must be an integer in the range [1, 31]
qc.setwmi x5, 0, 12(x12)

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be a multiple of 4 bytes in the range [0, 124]
qc.setwmi x5, 31, 98(x12)

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)
qc.setwmi x5, 31, 12(x12)


# CHECK: :[[@LINE+1]]:19: error: expected register
qc.lwm x7, x1, 24(20)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lwm x7, x1, 24

# CHECK: :[[@LINE+1]]:8: error: invalid operand for instruction
qc.lwm x0, x1, 24(x20)

# CHECK-PLUS: :[[@LINE+1]]:16: error: immediate must be a multiple of 4 bytes in the range [0, 124]
qc.lwm x7, x1, 46(x20)

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)
qc.lwm x7, x1, 24(x20)


# CHECK: :[[@LINE+1]]:19: error: expected register
qc.lwmi x13, 9, 4(23)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lwmi x13, 9, 4

# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.lwmi x0, 9, 4(x23)

# CHECK-PLUS: :[[@LINE+1]]:14: error: immediate must be an integer in the range [1, 31]
qc.lwmi x13, 44, 4(x23)

# CHECK-PLUS: :[[@LINE+1]]:14: error: immediate must be an integer in the range [1, 31]
qc.lwmi x13, 0, 4(x23)

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be a multiple of 4 bytes in the range [0, 124]
qc.lwmi x13, 9, 77(x23)

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)
qc.lwmi x13, 9, 4(x23)
