// RUN: not llvm-mc -filetype=asm -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

.section .text.f1,"ax",@progbits
// CHECK: {{.*}}.s:[[# @LINE+1]]:12: error: alignment must be a power of 2
.prefalign 3
