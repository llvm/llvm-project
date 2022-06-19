// RUN: llvm-mc %s --triple=arm-linux-gnueabihf -filetype=obj | llvm-objdump --no-show-raw-insn --triple=armv7 -d - | FileCheck %s

// Check that the architectural nop is only produced for subtargets that
// support it. This includes nop padding for alignment.
 .syntax unified
 .arch armv6
foo:
 mov r1, r0
 nop
 .p2align 4
 bx lr

 .arch armv7-a
bar:
 mov r1, r0
 nop
 .p2align 4
 bx lr

 .arch armv4t
baz:
 mov r1, r0
 nop
 .p2align 4
 bx lr

// CHECK: 00000000 <foo>:
// CHECK-NEXT:  0: mov     r1, r0
// CHECK-NEXT:  4: mov     r0, r0
// CHECK-NEXT:  8: mov     r0, r0
// CHECK-NEXT:  c: mov     r0, r0
// CHECK-NEXT: 10: bx      lr

// CHECK: 00000014 <bar>:
// CHECK-NEXT: 14: mov     r1, r0
// CHECK-NEXT: 18: nop
// CHECK-NEXT: 1c: nop
// CHECK-NEXT: 20: bx      lr

// CHECK: 00000024 <baz>:
// CHECK-NEXT: 24: mov     r1, r0
// CHECK-NEXT: 28: mov     r0, r0
// CHECK-NEXT: 2c: mov     r0, r0
// CHECK-NEXT: 30: bx      lr
