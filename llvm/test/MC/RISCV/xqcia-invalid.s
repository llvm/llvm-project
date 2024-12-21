# Xqcia - Qualcomm uC Arithmetic Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcia < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcia < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS %s

# CHECK: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.slasat x10, x3, 17

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.slasat x10, x3

# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.slasat x0, x3, x17

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.slasat x10, x0, x17

# CHECK: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.slasat x10, x3, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.slasat x10, x3, x17


# CHECK: :[[@LINE+1]]:21: error: invalid operand for instruction
qc.sllsat x23, x25, 27

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.sllsat x23, x25

# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.sllsat x0, x25, x27

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.sllsat x23, x0, x27

# CHECK: :[[@LINE+1]]:21: error: invalid operand for instruction
qc.sllsat x23, x25, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.sllsat x23, x25, x27


# CHECK: :[[@LINE+1]]:21: error: invalid operand for instruction
qc.addsat x17, x14, 7

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.addsat x17, x14

# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.addsat x0, x14, x7

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.addsat x17, x0, x7

# CHECK: :[[@LINE+1]]:21: error: invalid operand for instruction
qc.addsat x17, x14, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.addsat x17, x14, x7


# CHECK: :[[@LINE+1]]:21: error: invalid operand for instruction
qc.addusat x8, x18, 28

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.addusat x8, x18

# CHECK: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.addusat x0, x18, x28

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.addusat x8, x0, x28

# CHECK: :[[@LINE+1]]:21: error: invalid operand for instruction
qc.addusat x8, x18, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.addusat x8, x18, x28


# CHECK: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.subsat x22, x2, 12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.subsat x22, x2

# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.subsat x0, x2, x12

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.subsat x22, x0, x12

# CHECK: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.subsat x22, x2, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.subsat x22, x2, x12


# CHECK: :[[@LINE+1]]:21: error: invalid operand for instruction
qc.subusat x9, x14, 17

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.subusat x9, x14

# CHECK: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.subusat x0, x14, x17

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.subusat x9, x0, x17

# CHECK: :[[@LINE+1]]:21: error: invalid operand for instruction
qc.subusat x9, x14, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.subusat x9, x14, x17


# CHECK: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.wrap x3, x30, 23

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.wrap x3, x30

# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.wrap x0, x30, x23

# CHECK: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.wrap x3, x30, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.wrap x3, x30, x23


# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.wrapi x0, 12, 2047

# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.wrapi x0, x12, 2047

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.wrapi x6, x0, 2047

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.wrapi x6, x12

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be an integer in the range [0, 2047]
qc.wrapi x6, x12, 2048

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.wrapi x6, x12, 2047


# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.norm x3, 7

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.norm x3

# CHECK: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.norm x0, x7

# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.norm x3, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.norm x3, x7


# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.normu x11, 17

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.normu x11

# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.normu x0, x17

# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.normu x11, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.normu x11, x17


# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.normeu x26, 31

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.normeu x26

# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.normeu x0, x31

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.normeu x26, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.normeu x26, x31
