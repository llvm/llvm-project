# Xqcilo - Qualcomm uC Large Offset Load Store extension
# RUN: not llvm-mc %s -triple=riscv32 -mattr=+xqcilo,+xqcili \
# RUN:     2>&1 | FileCheck -check-prefixes=CHECK-ENABLED %s
# RUN: not llvm-mc %s -triple=riscv32 -mattr=-xqcilo \
# RUN:     2>&1 | FileCheck -check-prefixes=CHECK-DISABLED %s

# CHECK-ENABLED: [[@LINE+4]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-ENABLED: [[@LINE+3]]:13: note: operand must be a bare symbol name
# CHECK-ENABLED: [[@LINE+2]]:19: note: too few operands for instruction
# CHECK-DISABLED: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.e.lb a0, 0xf000

# CHECK-ENABLED: [[@LINE+4]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-ENABLED: [[@LINE+3]]:13: note: operand must be a bare symbol name
# CHECK-ENABLED: [[@LINE+2]]:19: note: too few operands for instruction
# CHECK-DISABLED: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.e.lb a0, 0xf000

# CHECK-ENABLED: [[@LINE+4]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-ENABLED: [[@LINE+3]]:14: note: operand must be a bare symbol name
# CHECK-ENABLED: [[@LINE+2]]:20: note: too few operands for instruction
# CHECK-DISABLED: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.e.lbu a0, 0xf000

# CHECK-ENABLED: [[@LINE+4]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-ENABLED: [[@LINE+3]]:13: note: operand must be a bare symbol name
# CHECK-ENABLED: [[@LINE+2]]:19: note: too few operands for instruction
# CHECK-DISABLED: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.e.lh a0, 0xf000

# CHECK-ENABLED: [[@LINE+4]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-ENABLED: [[@LINE+3]]:14: note: operand must be a bare symbol name
# CHECK-ENABLED: [[@LINE+2]]:20: note: too few operands for instruction
# CHECK-DISABLED: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.e.lhu a0, 0xf000

# CHECK-ENABLED: [[@LINE+4]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-ENABLED: [[@LINE+3]]:13: note: operand must be a bare symbol name
# CHECK-ENABLED: [[@LINE+2]]:19: note: too few operands for instruction
# CHECK-DISABLED: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.e.lw a0, 0xf000

# CHECK-ENABLED: [[@LINE+2]]:13: error: operand must be a bare symbol name
# CHECK-DISABLED: [[@LINE+1]]:1: error: invalid instruction
qc.e.sb a0, 0xf000, t0

# CHECK-ENABLED: [[@LINE+2]]:13: error: operand must be a bare symbol name
# CHECK-DISABLED: [[@LINE+1]]:1: error: invalid instruction
qc.e.sh a0, 0xf000, t0

# CHECK-ENABLED: [[@LINE+2]]:13: error: operand must be a bare symbol name
# CHECK-DISABLED: [[@LINE+1]]:1: error: invalid instruction
qc.e.sw a0, 0xf000, t0

# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lb a0, undefined
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lbu a0, undefined
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lh a0, undefined
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lhu a0, undefined
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.lw a0, undefined
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.sb a0, undefined, t0
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.sh a0, undefined, t0
# CHECK-DISABLED: [[@LINE+1]]:1: error: instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)
qc.e.sw a0, undefined, t0

# CHECK-ENABLED: [[@LINE+2]]:22: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: invalid instruction
qc.e.sb a0, undefined

# CHECK-ENABLED: [[@LINE+2]]:22: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: invalid instruction
qc.e.sh a0, undefined

# CHECK-ENABLED: [[@LINE+2]]:22: error: too few operands for instruction
# CHECK-DISABLED: [[@LINE+1]]:1: error: invalid instruction
qc.e.sw a0, undefined
