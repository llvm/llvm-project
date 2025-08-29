// RUN: llvm-mc -filetype=obj -triple=thumbv7a-linux-gnueabi %s -o %t
// RUN: llvm-objdump -dr %t --triple=thumbv7a | FileCheck %s

// CHECK:      f7ff fffe     bl      {{.*}}                  @ imm = #-0x4
// CHECK-NEXT:               00000000:  R_ARM_THM_CALL       foo
// CHECK-NEXT: f7ff effe     blx     {{.*}}                  @ imm = #-0x4
// CHECK-NEXT:               00000004:  R_ARM_THM_CALL       callee_thumb_low
// CHECK-NEXT: bf00          nop
// CHECK-NEXT: f7ff effe     blx     {{.*}}                  @ imm = #-0x4
// CHECK-NEXT:               0000000a:  R_ARM_THM_CALL       callee_thumb_high
// CHECK-NEXT: 4770          bx      lr

 .syntax unified
 .section .text, "ax",%progbits
 .thumb
 bl foo

 blx   callee_thumb_low
 nop
 blx   callee_thumb_high
 bx lr
