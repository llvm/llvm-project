# Xqcili - Qualcomm uC Load Large Immediate Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+xqcili < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-xqcili < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-MINUS %s

# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:9: error: register must be a GPR
qc.e.li 9, 33554432

# CHECK-PLUS: :[[@LINE+2]]:11: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.e.li x9

# CHECK-PLUS: :[[@LINE+3]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-PLUS: :[[@LINE+2]]:13: note: immediate must be an integer in the range [-2147483648, 4294967295]
# CHECK-PLUS: :[[@LINE+1]]:13: note: immediate must be an integer in the range [-2147483648, 4294967295]
qc.e.li x9, 4294967296

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension)
qc.e.li x9, 4294967295


# CHECK-PLUS: :[[@LINE+2]]:7: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.li x0, 114514

# CHECK-PLUS: :[[@LINE+2]]:10: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.li x10

# CHECK-PLUS: :[[@LINE+1]]:12: error: operand must be a symbol with a %qc.abs20 specifier or an integer in the range [-524288, 524287]
qc.li x10, 33554432

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension)
qc.li x10, 114514
