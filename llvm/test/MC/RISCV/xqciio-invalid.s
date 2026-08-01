# Xqciio - Qualcomm uC External Input Output Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+xqciio < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-xqciio < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS %s

# CHECK: :[[@LINE+1]]:18: error: expected register
qc.outw x5, 2048(10)

# CHECK-PLUS: :[[@LINE+3]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-PLUS: :[[@LINE+2]]:13: note: invalid operand for instruction
# CHECK-PLUS: :[[@LINE+1]]:13: note: immediate must be a multiple of 4 bytes in the range [0, 16380]
qc.outw x5, x10

# CHECK-MINUS: :[[@LINE+3]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-MINUS: :[[@LINE+2]]:13: note: invalid operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:13: note: immediate must be a multiple of 4 bytes in the range [0, 16380]
qc.outw x5, x10

# CHECK-PLUS: :[[@LINE+1]]:13: error: immediate must be a multiple of 4 bytes in the range [0, 16380]
qc.outw x5, 4099(x10)

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciio' (Qualcomm uC External Input Output Extension)
qc.outw x5, 2048(x10)


# CHECK: :[[@LINE+1]]:19: error: expected register
qc.inw x23, 16380(17)

# CHECK-PLUS: :[[@LINE+3]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-PLUS: :[[@LINE+2]]:13: note: invalid operand for instruction
# CHECK-PLUS: :[[@LINE+1]]:13: note: immediate must be a multiple of 4 bytes in the range [0, 16380]
qc.inw x23, x17

# CHECK-MINUS: :[[@LINE+3]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-MINUS: :[[@LINE+2]]:13: note: invalid operand for instruction
# CHECK-MINUS: :[[@LINE+1]]:13: note: immediate must be a multiple of 4 bytes in the range [0, 16380]
qc.inw x23, x17

# CHECK-PLUS: :[[@LINE+2]]:8: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:8: error: register must be a GPR excluding zero (x0)
qc.inw x0, 16380(x17)

# CHECK-PLUS: :[[@LINE+1]]:13: error: immediate must be a multiple of 4 bytes in the range [0, 16380]
qc.inw x23, 16384(x17)

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciio' (Qualcomm uC External Input Output Extension)
qc.inw x23, 16380(x17)
