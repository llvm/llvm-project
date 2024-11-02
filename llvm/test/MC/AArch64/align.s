// RUN: llvm-mc -filetype=obj -triple aarch64-none-eabi %s | llvm-objdump -d - | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple aarch64_be-none-eabi %s | llvm-objdump -d - | FileCheck %s

// CHECK:   0: d2800000   mov     x0, #0
// CHECK:   4: d2800000   mov     x0, #0
// CHECK:   8: d503201f   nop
// CHECK:   c: d503201f   nop
// CHECK:  10: d2800000   mov     x0, #0

       .text
       mov x0, #0
       mov x0, #0
       .p2align 4
       mov x0, #0
