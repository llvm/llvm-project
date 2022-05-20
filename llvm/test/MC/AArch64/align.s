// RUN: llvm-mc -filetype=obj -triple aarch64-none-eabi %s | llvm-objdump -d - | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple aarch64_be-none-eabi %s | llvm-objdump -d - | FileCheck %s

// CHECK:   0: 00 00 80 d2   mov     x0, #0
// CHECK:   4: 00 00 80 d2   mov     x0, #0
// CHECK:   8: 1f 20 03 d5   nop
// CHECK:   c: 1f 20 03 d5   nop
// CHECK:  10: 00 00 80 d2   mov     x0, #0

       .text
       mov x0, #0
       mov x0, #0
       .p2align 4
       mov x0, #0
