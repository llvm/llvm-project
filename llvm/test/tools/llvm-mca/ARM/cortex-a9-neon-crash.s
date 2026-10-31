# RUN: llvm-mca -mtriple=armv7-none-linux-gnueabihf -mcpu=cortex-a9 -iterations=2 < %s | FileCheck %s --check-prefix=TWO
# RUN: llvm-mca -mtriple=armv7-none-linux-gnueabihf -mcpu=cortex-a9 -iterations=100 < %s | FileCheck %s --check-prefix=HUNDRED

# Reproducer from issue #225087.
.syntax unified
vmov.i8 q8, #85
vmov.i8 q9, #51
vmov.i8 q10, #15
vmov.i32 q11, #255
vmov.i32 q12, #65280
lsr r5, r7, #5
mov r7, #0
add r5, r1, r5, lsl #2
ldrne r7, [r5, #-4]
vld1.32 {d26-d27}, [r5]
add r5, r2, r6
sub r6, r6, #64
vmov.32 d29[1], r7
str r7, [r0, #4]

# TWO: Iterations:        2
# TWO-NEXT: Instructions:      28
# HUNDRED: Iterations:        100
# HUNDRED-NEXT: Instructions:      1400
