@ RUN: llvm-mc < %s -triple armv7-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -r - | FileCheck %s

  .code  16
  .thumb_func
thumb_caller:
  beq.w internal_arm_fn
  beq.w global_arm_fn
  beq.w global_thumb_fn
  beq.w internal_thumb_fn

  .type  internal_arm_fn,%function
  .code  32
internal_arm_fn:
  bx  lr

  .globl  global_arm_fn
  .type  global_arm_fn,%function
  .code  32
global_arm_fn:
  bx  lr

  .type  internal_thumb_fn,%function
  .code  16
  .thumb_func
internal_thumb_fn:
  bx  lr

  .globl  global_thumb_fn
  .type  global_thumb_fn,%function
  .code  16
  .thumb_func
global_thumb_fn:
  bx  lr

@ CHECK: Section (3) .rel.text
@ CHECK-NEXT: 0x0 R_ARM_THM_JUMP19 internal_arm_fn
@ CHECK-NEXT: 0x4 R_ARM_THM_JUMP19 global_arm_fn
@ CHECK-NEXT: 0x8 R_ARM_THM_JUMP19 global_thumb_fn
@ CHECK-NEXT: }
