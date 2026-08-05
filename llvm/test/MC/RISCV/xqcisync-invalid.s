# Xqcisync - Qualcomm uC Sync Delay Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+xqcisync < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-xqcisync < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-MINUS %s

# CHECK-PLUS: :[[@LINE+1]]:12: error: immediate must be an integer in the range [1, 31]
qc.c.delay 34

# CHECK-PLUS: :[[@LINE+2]]:16: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.delay 11, 12

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.delay

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)
qc.c.delay   10


# CHECK-PLUS: :[[@LINE+1]]:9: error: immediate must be an integer in the range [0, 31]
qc.sync 45

# CHECK-PLUS: :[[@LINE+2]]:13: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.sync 22, x4

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.sync

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)
qc.sync 8


# CHECK-PLUS: :[[@LINE+1]]:10: error: immediate must be an integer in the range [0, 31]
qc.syncr 56

# CHECK-PLUS: :[[@LINE+2]]:14: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.syncr 31, 45

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.syncr

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)
qc.syncr   23


# CHECK-PLUS: :[[@LINE+1]]:11: error: immediate must be an integer in the range [0, 31]
qc.syncwf 88

# CHECK-PLUS: :[[@LINE+2]]:14: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.syncwf 5, 44

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.syncwf

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)
qc.syncwf  31


# CHECK-PLUS: :[[@LINE+1]]:11: error: immediate must be an integer in the range [0, 31]
qc.syncwl 99

# CHECK-PLUS: :[[@LINE+2]]:15: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.syncwl 11, x10

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.syncwl

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)
qc.syncwl  1


# CHECK-PLUS: :[[@LINE+1]]:11: error: immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31
qc.c.sync 45

# CHECK-PLUS: :[[@LINE+2]]:15: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.sync 31, x4

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.sync

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)
qc.c.sync 8


# CHECK-PLUS: :[[@LINE+1]]:12: error: immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31
qc.c.syncr 56

# CHECK-PLUS: :[[@LINE+2]]:16: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.syncr 31, 45

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.syncr

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)
qc.c.syncr   8


# CHECK-PLUS: :[[@LINE+1]]:13: error: immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31
qc.c.syncwf 88

# CHECK-PLUS: :[[@LINE+2]]:16: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.syncwf 8, 44

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.syncwf

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)
qc.c.syncwf  31


# CHECK-PLUS: :[[@LINE+1]]:13: error: immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31
qc.c.syncwl 99

# CHECK-PLUS: :[[@LINE+2]]:17: error: unexpected extra operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.syncwl 15, x10

# CHECK-PLUS: :[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: :[[@LINE+1]]:1: error: invalid instruction
qc.c.syncwl

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)
qc.c.syncwl  1
