# RUN: not llvm-mc -triple armv7 %s 2>&1| FileCheck %s

.arm

ADC r0, r1, #0xFFFFFEEE
# CHECK: error: invalid operand for instruction
ADC r0, r1, #0xABFEABFF
# CHECK: error: invalid operand for instruction
ADC r0, r1, #0xFFFFFE02
# CHECK: error: invalid operand for instruction

ADD.W r0, r0, #0xFF01FF01
# CHECK: error: invalid operand for instruction
