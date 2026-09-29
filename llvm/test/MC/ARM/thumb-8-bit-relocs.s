@ RUN: llvm-mc -triple thumbv6m-eabi -o - %s | FileCheck %s
@ RUN: llvm-mc -triple thumbv6m-eabi -filetype obj -o - %s | llvm-readobj -r - \
@ RUN:   | FileCheck -check-prefix CHECK-RELOCATIONS %s
@ RUN: llvm-mc -triple thumbv7m-eabi -o - %s | FileCheck %s
@ RUN: llvm-mc -triple thumbv7m-eabi -filetype obj -o - %s | llvm-readobj -r - \
@ RUN:   | FileCheck -check-prefix CHECK-RELOCATIONS %s

.syntax unified

.type function,%function
function:
  bx lr

.global external
.type external,%function

.type test,%function
test:
  movs r3, :upper8_15:function
  adds r3, :upper0_7:function
  adds r3, :lower8_15:function
  adds r3, :lower0_7:function

@ CHECK-LABEL: test:
@ CHECK:  movs r3, :upper8_15:function
@ CHECK:  adds r3, :upper0_7:function
@ CHECK:  adds r3, :lower8_15:function
@ CHECK:  adds r3, :lower0_7:function

@ CHECK-RELOCATIONS: Relocations [
@ CHECK-RELOCATIONS:     0x2 R_ARM_THM_ALU_ABS_G3 function
@ CHECK-RELOCATIONS-NEXT:     0x4 R_ARM_THM_ALU_ABS_G2_NC function
@ CHECK-RELOCATIONS-NEXT:     0x6 R_ARM_THM_ALU_ABS_G1_NC function
@ CHECK-RELOCATIONS-NEXT:     0x8 R_ARM_THM_ALU_ABS_G0_NC function
@ CHECK-RELOCATIONS: ]
