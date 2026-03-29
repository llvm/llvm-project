# Xqcilo - Qualcomm uC Large Offset Load Store extension
# RUN: not llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilo \
# RUN:     2>&1 | FileCheck -check-prefixes=CHECK-ENABLED %s
# RUN: not llvm-mc %s -triple=riscv32 -mattr=-experimental-xqcilo \
# RUN:     2>&1 | FileCheck -check-prefixes=CHECK-DISABLED %s

# CHECK-ENABLED: [[@LINE+2]]:1: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: too few operands for instruction
qc.e.lb a0, 0xf000

# CHECK-ENABLED: [[@LINE+2]]:1: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: too few operands for instruction
qc.e.lb a0, 0xf000

# CHECK-ENABLED: [[@LINE+2]]:1: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: too few operands for instruction
qc.e.lbu a0, 0xf000

# CHECK-ENABLED: [[@LINE+2]]:1: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: too few operands for instruction
qc.e.lh a0, 0xf000

# CHECK-ENABLED: [[@LINE+2]]:1: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: too few operands for instruction
qc.e.lhu a0, 0xf000

# CHECK-ENABLED: [[@LINE+2]]:1: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: too few operands for instruction
qc.e.lw a0, 0xf000

# CHECK-ENABLED: [[@LINE+2]]:21: error: invalid operand for instruction
# CHECK-DISABLED: [[@LINE+1]]:21: error: invalid operand for instruction
qc.e.sb a0, 0xf000, t0

# CHECK-ENABLED: [[@LINE+2]]:21: error: invalid operand for instruction
# CHECK-DISABLED: [[@LINE+1]]:21: error: invalid operand for instruction
qc.e.sh a0, 0xf000, t0

# CHECK-ENABLED: [[@LINE+2]]:21: error: invalid operand for instruction
# CHECK-DISABLED: [[@LINE+1]]:21: error: invalid operand for instruction
qc.e.sw a0, 0xf000, t0

# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lb a0, undefined
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lbu a0, undefined
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lh a0, undefined
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lhu a0, undefined
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lw a0, undefined
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.sb a0, undefined, t0
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.sh a0, undefined, t0
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.sw a0, undefined, t0

# CHECK-ENABLED: [[@LINE+2]]:1: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: too few operands for instruction
qc.e.sb a0, undefined

# CHECK-ENABLED: [[@LINE+2]]:1: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: too few operands for instruction
qc.e.sh a0, undefined

# CHECK-ENABLED: [[@LINE+2]]:1: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: too few operands for instruction
qc.e.sw a0, undefined
