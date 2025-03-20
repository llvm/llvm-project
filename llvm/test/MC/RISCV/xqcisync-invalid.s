# Xqcisync - Qualcomm uC Synchronization And Delay Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcisync < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS,CHECK-IMM %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcisync < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS,CHECK-EXT %s

# CHECK-PLUS: :[[@LINE+2]]:12: error: immediate must be an integer in the range [1, 31]
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.c.delay 0

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.delay

# CHECK-IMM: :[[@LINE+1]]:12: error: immediate must be an integer in the range [1, 31]
qc.c.delay 32

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Synchronization And Delay Extension)
qc.c.delay 5

# CHECK-PLUS: :[[@LINE+2]]:11: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.c.sync 8

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.sync

# CHECK-IMM: :[[@LINE+1]]:11: error: immediate must be an integer in the range [0, 7]
qc.c.sync -1

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Synchronization And Delay Extension)
qc.c.sync 3

# CHECK-PLUS: :[[@LINE+2]]:12: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.c.syncr 10

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.syncr

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Synchronization And Delay Extension)
qc.c.syncr 3

# CHECK-PLUS: :[[@LINE+2]]:13: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.c.syncwf 8

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.syncwf

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Synchronization And Delay Extension)
qc.c.syncwf 5

# CHECK-PLUS: :[[@LINE+2]]:13: error: immediate must be an integer in the range [0, 7]
# CHECK-MINUS: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.c.syncwl 8

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.syncwl

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Synchronization And Delay Extension)
qc.c.syncwl 7

# CHECK-PLUS: :[[@LINE+2]]:9: error: immediate must be an integer in the range [0, 31]
# CHECK-MINUS: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.sync 32

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.sync

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Synchronization And Delay Extension)
qc.sync 10

# CHECK-PLUS: :[[@LINE+2]]:10: error: immediate must be an integer in the range [0, 31]
# CHECK-MINUS: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.syncr -1

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.syncr

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Synchronization And Delay Extension)
qc.syncr 10

# CHECK-PLUS: :[[@LINE+2]]:11: error: immediate must be an integer in the range [0, 31]
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.syncwf 33

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.syncwf

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Synchronization And Delay Extension)
qc.syncwf 10

# CHECK-PLUS: :[[@LINE+2]]:11: error: immediate must be an integer in the range [0, 31]
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.syncwl -1

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.syncwl

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisync' (Qualcomm uC Synchronization And Delay Extension)
qc.syncwl 10
