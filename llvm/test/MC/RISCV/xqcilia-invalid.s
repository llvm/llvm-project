# Xqcilia - Qualcomm uC Large Immediate Arithmetic extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcilia < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS,CHECK-IMM %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcilia < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS,CHECK-EXT %s

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.e.addai 9, 33554432

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.addai x9

# CHECK-IMM: :[[@LINE+1]]:16: error: immediate must be an integer in the range [-2147483648, 4294967295]
qc.e.addai x9, 20485546494

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)
qc.e.addai x9, 33554432


# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.e.addi x10, 9, 554432

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.addi x10, x9

# CHECK-IMM: :[[@LINE+1]]:20: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.addi x10, x9, 335544312

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)
qc.e.addi x10, x9, 554432


# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.e.andai 9, 33554432

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.andai x9

# CHECK-IMM: :[[@LINE+1]]:16: error: immediate must be an integer in the range [-2147483648, 4294967295]
qc.e.andai x9, 20494437494

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)
qc.e.andai x9, 33554432


# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.e.andi x10, 9, 554432

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.andi x10, x9

# CHECK-IMM: :[[@LINE+1]]:20: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.andi x10, x9, 335544312

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)
qc.e.andi x10, x9, 554432


# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.e.orai 9, 33554432

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.orai x9

# CHECK-IMM: :[[@LINE+1]]:15: error: immediate must be an integer in the range [-2147483648, 4294967295]
qc.e.orai x9, 20494437494

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)
qc.e.orai x9, 33554432


# CHECK-PLUS: :[[@LINE+2]]:15: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.e.ori x10, 9, 554432

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.ori x10, x9

# CHECK-IMM: :[[@LINE+1]]:19: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.ori x10, x9, 335544312

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)
qc.e.ori x10, x9, 554432



# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.e.xorai 9, 33554432

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.xorai x9

# CHECK-IMM: :[[@LINE+1]]:16: error: immediate must be an integer in the range [-2147483648, 4294967295]
qc.e.xorai x9, 20494437494

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)
qc.e.xorai x9, 33554432


# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.e.xori x10, 9, 554432

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.xori x10, x9

# CHECK-IMM: :[[@LINE+1]]:20: error: immediate must be an integer in the range [-33554432, 33554431]
qc.e.xori x10, x9, 335544312

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)
qc.e.xori x10, x9, 554432
