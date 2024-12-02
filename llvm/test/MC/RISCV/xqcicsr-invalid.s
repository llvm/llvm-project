# Xqcicsr - Qualcomm uC CSR Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcicsr < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcicsr < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS %s

# CHECK: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.csrrwr x10, x5, x0

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.csrrwr x10, x5

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicsr' (Qualcomm uC CSR Extension)
qc.csrrwr x10, x5, x20


# CHECK: :[[@LINE+1]]:21: error: invalid operand for instruction
qc.csrrwri x20, 31, x0

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be an integer in the range [0, 31]
qc.csrrwri x20, 45, x12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.csrrwri x20, 23

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcicsr' (Qualcomm uC CSR Extension)
qc.csrrwri x30, 31, x12
