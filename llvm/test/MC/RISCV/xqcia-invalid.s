# Xqcia - Qualcomm uC Arithmetic Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+xqcia < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-xqcia < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-MINUS %s

# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.shlsat x10, x3, 17

# CHECK-PLUS: :[[@LINE+2]]:18: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.shlsat x10, x3

# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.shlsat x0, x3, x17

# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.shlsat x10, x0, x17

# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.shlsat x10, x3, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.shlsat x10, x3, x17


# CHECK-PLUS: :[[@LINE+2]]:22: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.shlusat x23, x25, 27

# CHECK-PLUS: :[[@LINE+2]]:20: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.shlusat x23, x25

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.shlusat x0, x25, x27

# CHECK-PLUS: :[[@LINE+2]]:17: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.shlusat x23, x0, x27

# CHECK-PLUS: :[[@LINE+2]]:22: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.shlusat x23, x25, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.shlusat x23, x25, x27


# CHECK-PLUS: :[[@LINE+2]]:21: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.addsat x17, x14, 7

# CHECK-PLUS: :[[@LINE+2]]:19: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.addsat x17, x14

# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.addsat x0, x14, x7

# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.addsat x17, x0, x7

# CHECK-PLUS: :[[@LINE+2]]:21: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.addsat x17, x14, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.addsat x17, x14, x7


# CHECK-PLUS: :[[@LINE+2]]:21: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.addusat x8, x18, 28

# CHECK-PLUS: :[[@LINE+2]]:19: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.addusat x8, x18

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.addusat x0, x18, x28

# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.addusat x8, x0, x28

# CHECK-PLUS: :[[@LINE+2]]:21: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.addusat x8, x18, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.addusat x8, x18, x28


# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.subsat x22, x2, 12

# CHECK-PLUS: :[[@LINE+2]]:18: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.subsat x22, x2

# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.subsat x0, x2, x12

# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.subsat x22, x0, x12

# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.subsat x22, x2, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.subsat x22, x2, x12


# CHECK-PLUS: :[[@LINE+2]]:21: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.subusat x9, x14, 17

# CHECK-PLUS: :[[@LINE+2]]:19: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.subusat x9, x14

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.subusat x0, x14, x17

# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.subusat x9, x0, x17

# CHECK-PLUS: :[[@LINE+2]]:21: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.subusat x9, x14, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.subusat x9, x14, x17


# CHECK-PLUS: :[[@LINE+2]]:18: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.wrap x3, x30, 23

# CHECK-PLUS: :[[@LINE+2]]:16: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.wrap x3, x30

# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.wrap x0, x30, x23

# CHECK-PLUS: :[[@LINE+2]]:18: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.wrap x3, x30, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.wrap x3, x30, x23


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:10: error: register must be a GPR excluding zero (x0)
qc.wrapi x0, 12, 2047

# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.wrapi x0, x12, 2047

# CHECK-PLUS: :[[@LINE+2]]:14: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.wrapi x6, x0, 2047

# CHECK-PLUS: :[[@LINE+2]]:17: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.wrapi x6, x12

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be an integer in the range [0, 2047]
qc.wrapi x6, x12, 2048

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.wrapi x6, x12, 2047


# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.norm x3, 7

# CHECK-PLUS: :[[@LINE+2]]:11: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.norm x3

# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.norm x0, x7

# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.norm x3, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.norm x3, x7


# CHECK-PLUS: :[[@LINE+2]]:15: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.normu x11, 17

# CHECK-PLUS: :[[@LINE+2]]:13: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.normu x11

# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.normu x0, x17

# CHECK-PLUS: :[[@LINE+2]]:15: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.normu x11, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.normu x11, x17


# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.normeu x26, 31

# CHECK-PLUS: :[[@LINE+2]]:14: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.normeu x26

# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.normeu x0, x31

# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.normeu x26, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)
qc.normeu x26, x31
