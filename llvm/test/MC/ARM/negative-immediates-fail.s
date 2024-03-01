# RUN: not llvm-mc -triple armv7 %s 2>&1| FileCheck %s

.arm

ADC r0, r1, #0xFFFFFEEE
# CHECK: error: invalid instruction, any one of the following would fix this:
# CHECK: note: too many operands for instruction
# CHECK: note: operand must be a register in range [r0, r15]
ADC r0, r1, #0xABFEABFF
# CHECK: error: invalid instruction, any one of the following would fix this:
# CHECK: note: too many operands for instruction
# CHECK: note: operand must be a register in range [r0, r15]
ADC r0, r1, #0xFFFFFE02
# CHECK: error: invalid instruction, any one of the following would fix this:
# CHECK: note: too many operands for instruction
# CHECK: note: operand must be a register in range [r0, r15]

ADD.W r0, r0, #0xFF01FF01
# CHECK: error: invalid instruction, any one of the following would fix this:
# CHECK: note: too many operands for instruction
# CHECK: note: operand must be a register in range [r0, r15]

ORR r0, r1, #0xFFFFFF00
# CHECK: error: invalid instruction, any one of the following would fix this:
# CHECK: note: too many operands for instruction
# CHECK: note: operand must be a register in range [r0, r15]
# CHECK: note: instruction requires: thumb2
ORN r0, r1, #0xFFFFFF00
# CHECK: error: instruction requires: thumb2
