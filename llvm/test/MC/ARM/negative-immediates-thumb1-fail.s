# RUN: not llvm-mc -triple thumbv7 -mcpu=cortex-m0 %s 2>&1 | FileCheck %s

.thumb

ADDs r1, r0, #0xFFFFFFF5
# CHECK: error: instruction requires: arm-mode

ADDs r0, #0xFFFFFEFF
# CHECK: error: invalid operand for instruction

SUBs r1, r0, #0xFFFFFFF5
# CHECK: error: instruction requires: arm-mode

SUBs r0, #0xFFFFFEFF
# CHECK: error: invalid operand for instruction
