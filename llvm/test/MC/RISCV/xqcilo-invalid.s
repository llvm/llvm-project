# Xqcilo - Qualcomm uC Extension Large Offset Load Store extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcilo < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-IMM %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcilo < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-EXT %s

# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.e.lb 11, 12(x10)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.lb x11

# CHECK-IMM: :[[@LINE+1]]:14: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.lb x11, 33445562212(x10)

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lb x11, 12(x10)


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.e.lbu 11, 12(x10)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.lbu x11

# CHECK-IMM: :[[@LINE+1]]:15: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.lbu x11, 33445562212(x10)

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lbu x11, 12(x10)


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.e.lh 11, 12(x10)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.lh x11

# CHECK-IMM: :[[@LINE+1]]:14: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.lh x11, 33445562212(x10)

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lh x11, 12(x10)


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.e.lhu 11, 12(x10)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.lhu x11

# CHECK-IMM: :[[@LINE+1]]:15: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.lhu x11, 33445562212(x10)

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lhu x11, 12(x10)


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.e.lw 11, 12(x10)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.lw x11

# CHECK-IMM: :[[@LINE+1]]:14: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.lw x11, 33445562212(x10)

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lw x11, 12(x10)


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.e.sb 11, 12(x10)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.sb x11

# CHECK-IMM: :[[@LINE+1]]:14: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.sb x11, 33445562212(x10)

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.sb x11, 12(x10)


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.e.sh 11, 12(x10)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.sh x11

# CHECK-IMM: :[[@LINE+1]]:14: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.sh x11, 33445562212(x10)

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.sh x11, 12(x10)


# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.e.sw 11, 12(x10)

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.sw x11

# CHECK-IMM: :[[@LINE+1]]:14: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.sw x11, 33445562212(x10)

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.sw x11, 12(x10)
