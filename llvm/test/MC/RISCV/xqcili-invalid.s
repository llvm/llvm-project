# Xqcili - Qualcomm uC Load Large Immediate Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcili < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS,CHECK-IMM %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcili < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS,CHECK-EXT %s

# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.e.li 9, 33554432

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.li x9

# CHECK-IMM: :[[@LINE+1]]:13: error: immediate must be an integer in the range [-2147483648, 4294967295]
qc.e.li x9, 4294967296

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension)
qc.e.li x9, 4294967295


# CHECK-PLUS: :[[@LINE+2]]:7: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:7: error: invalid operand for instruction
qc.li x0, 114514

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.li x10

# CHECK-IMM: :[[@LINE+1]]:12: error: immediate must be an integer in the range [-524288, 524287]
qc.li x10, 33554432

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension)
qc.li x10, 114514
