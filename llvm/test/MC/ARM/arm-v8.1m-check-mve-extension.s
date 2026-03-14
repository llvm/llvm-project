// RUN: llvm-mc -triple thumbv8.1m.main-none-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

.arch_extension mve.fp
vsub.f32	q0, q0, q1
// CHECK: vsub.f32	q0, q0, q1

