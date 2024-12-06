# Xqcisls - Qualcomm uC Scaled Load Store Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcisls < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcisls < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS %s

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.lrb x5, x2, x0, 4

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lrb x5, x2, x4

# CHECK-PLUS: :[[@LINE+1]]:20: error: immediate must be an integer in the range [0, 7]
qc.lrb x5, x2, x4, 12

# CHECK: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.lrb x5, 2, x4, 4

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.lrb x5, x2, x4, 4


# CHECK: :[[@LINE+1]]:17: error: invalid operand for instruction
qc.lrh x1, x12, x0, 2

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lrh x1, x12, x6

# CHECK-PLUS: :[[@LINE+1]]:21: error: immediate must be an integer in the range [0, 7]
qc.lrh x1, x12, x6, 22

# CHECK: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.lrh x1, 12, x6, 2

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.lrh x1, x12, x6, 2


# CHECK: :[[@LINE+1]]:17: error: invalid operand for instruction
qc.lrw x15, x7, x0, 1

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lrw x15, x7, x14

# CHECK-PLUS: :[[@LINE+1]]:22: error: immediate must be an integer in the range [0, 7]
qc.lrw x15, x7, x14, 11

# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.lrw x15, 7, x14, 1

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.lrw x15, x7, x14, 1


# CHECK: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.lrbu x9, x11, x0, 7

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lrbu x9, x11, x4

# CHECK-PLUS: :[[@LINE+1]]:22: error: immediate must be an integer in the range [0, 7]
qc.lrbu x9, x11, x4, 37

# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.lrbu x9, 11, x4, 7

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.lrbu x9, x11, x4, 7


# CHECK: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.lrhu x16, x6, x0, 4

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.lrhu x16, x6, x10

# CHECK-PLUS: :[[@LINE+1]]:23: error: immediate must be an integer in the range [0, 7]
qc.lrhu x16, x6, x10, 44

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.lrhu x16, 6, x10, 4

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.lrhu x16, x6, x10, 4


# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.srb x0, x2, x0, 3

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.srb x0, x2, x8

# CHECK-PLUS: :[[@LINE+1]]:20: error: immediate must be an integer in the range [0, 7]
qc.srb x0, x2, x8, 93

# CHECK: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.srb x0, 2, x8, 3

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.srb x0, x2, x8, 3


# CHECK: :[[@LINE+1]]:17: error: invalid operand for instruction
qc.srh x13, x0, x0, 6

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.srh x13, x0, x20

# CHECK-PLUS: :[[@LINE+1]]:22: error: immediate must be an integer in the range [0, 7]
qc.srh x13, x0, x20, 76

# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.srh x13, 0, x20, 6

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.srh x13, x0, x20, 6


# CHECK: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.srw x17, x18, x0, 0

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.srw x17, x18, x19

# CHECK-PLUS: :[[@LINE+1]]:23: error: immediate must be an integer in the range [0, 7]
qc.srw x17, x18, x19, 10

# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.srw x17, 18, x19, 0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.srw x17, x18, x19, 0
