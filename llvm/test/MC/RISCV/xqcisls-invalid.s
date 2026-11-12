# Xqcisls - Qualcomm uC Scaled Load Store Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+xqcisls < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-xqcisls < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-MINUS %s

# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrb x5, x2, x0, 4

# CHECK-PLUS: :[[@LINE+2]]:18: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrb x5, x2, x4

# CHECK-PLUS: :[[@LINE+2]]:20: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrb x5, x2, x4, 12

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrb x5, 2, x4, 4

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.lrb x5, x2, x4, 4


# CHECK-PLUS: :[[@LINE+2]]:17: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrh x1, x12, x0, 2

# CHECK-PLUS: :[[@LINE+2]]:19: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrh x1, x12, x6

# CHECK-PLUS: :[[@LINE+2]]:21: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrh x1, x12, x6, 22

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrh x1, 12, x6, 2

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.lrh x1, x12, x6, 2


# CHECK-PLUS: :[[@LINE+2]]:17: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrw x15, x7, x0, 1

# CHECK-PLUS: :[[@LINE+2]]:20: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrw x15, x7, x14

# CHECK-PLUS: :[[@LINE+2]]:22: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrw x15, x7, x14, 11

# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrw x15, 7, x14, 1

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.lrw x15, x7, x14, 1


# CHECK-PLUS: :[[@LINE+2]]:18: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrbu x9, x11, x0, 7

# CHECK-PLUS: :[[@LINE+2]]:20: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrbu x9, x11, x4

# CHECK-PLUS: :[[@LINE+2]]:22: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrbu x9, x11, x4, 37

# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrbu x9, 11, x4, 7

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.lrbu x9, x11, x4, 7


# CHECK-PLUS: :[[@LINE+2]]:18: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrhu x16, x6, x0, 4

# CHECK-PLUS: :[[@LINE+2]]:21: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrhu x16, x6, x10

# CHECK-PLUS: :[[@LINE+2]]:23: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrhu x16, x6, x10, 44

# CHECK-PLUS: :[[@LINE+2]]:14: error: register must be a GPR
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.lrhu x16, 6, x10, 4

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.lrhu x16, x6, x10, 4


# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srb x0, x2, x0, 3

# CHECK-PLUS: :[[@LINE+2]]:18: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srb x0, x2, x8

# CHECK-PLUS: :[[@LINE+2]]:20: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srb x0, x2, x8, 93

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srb x0, 2, x8, 3

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.srb x0, x2, x8, 3


# CHECK-PLUS: :[[@LINE+2]]:17: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srh x13, x0, x0, 6

# CHECK-PLUS: :[[@LINE+2]]:20: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srh x13, x0, x20

# CHECK-PLUS: :[[@LINE+2]]:22: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srh x13, x0, x20, 76

# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srh x13, 0, x20, 6

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.srh x13, x0, x20, 6


# CHECK-PLUS: :[[@LINE+2]]:18: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srw x17, x18, x0, 0

# CHECK-PLUS: :[[@LINE+2]]:21: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srw x17, x18, x19

# CHECK-PLUS: :[[@LINE+2]]:23: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srw x17, x18, x19, 10

# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.srw x17, 18, x19, 0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)
qc.srw x17, x18, x19, 0
