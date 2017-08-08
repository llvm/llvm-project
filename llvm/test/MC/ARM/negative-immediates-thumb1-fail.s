# RUN: not llvm-mc -triple thumbv7 -mcpu=cortex-m0 %s 2>&1 | FileCheck %s

.thumb

ADDs r1, r0, #0xFFFFFFF5
# CHECK: error: instruction requires: arm-mode

ADDs r0, #0xFFFFFEFF
# CHECK: error: immediate operand must be in the range [0,255]

SUBs r1, r0, #0xFFFFFFF5
# CHECK: error: instruction requires: arm-mode

SUBs r0, #0xFFFFFEFF
# CHECK: error: immediate operand must be in the range [0,255]

ORRs r0, r1, #0xFFFFFF00
# CHECK: error: instruction requires: thumb2
ORNs r0, r1, #0xFFFFFF00
# CHECK: error: instruction requires: thumb2
