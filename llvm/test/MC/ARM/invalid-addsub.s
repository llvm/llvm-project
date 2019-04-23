@ RUN: not llvm-mc -triple thumbv7-apple-ios %s -o - 2>&1 | FileCheck %s

@ CHECK: error: source register must be sp if destination is sp
@ CHECK: error: source register must be sp if destination is sp
@ CHECK: error: source register must be sp if destination is sp
@ CHECK: error: source register must be sp if destination is sp
add sp, r5, #1
addw sp, r7, #4
add sp, r3, r2
add sp, r3, r5, lsl #3


@ CHECK: error: source register must be sp if destination is sp
@ CHECK: error: source register must be sp if destination is sp
@ CHECK: error: source register must be sp if destination is sp
@ CHECK: error: source register must be sp if destination is sp
sub sp, r5, #1
subw sp, r7, #4
sub sp, r3, r2
sub sp, r3, r5, lsl #3
