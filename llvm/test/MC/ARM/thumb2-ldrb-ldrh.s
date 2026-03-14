@ RUN: not llvm-mc -triple thumbv7a-none-eabi -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple thumbv7m-none-eabi -show-encoding < %s 2>&1 | FileCheck %s

@ Thumb2 LDRS?[BH] are not valid when Rt == PC (these encodings are used for
@ preload hints).
@ We don't check the actual error messages here as they are currently not very
@ helpful, see http://llvm.org/bugs/show_bug.cgi?id=21066.

@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
  ldrb    pc, [r0, #10]
  ldrb.w  pc, [r1, #10]
  ldrb    pc, [r2, #-5]
  ldrb    pc, [pc, #7]
  ldrb.w  pc, [pc, #7]

@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
  ldrsb   pc, [r3, #10]
  ldrsb.w pc, [r4, #10]
  ldrsb   pc, [r5, #-5]
  ldrsb   pc, [pc, #7]
  ldrsb.w pc, [pc, #7]

@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
  ldrh    pc, [r6, #10]
  ldrh.w  pc, [r7, #10]
  ldrh    pc, [r8, #-5]
  ldrh    pc, [pc, #7]
  ldrh.w  pc, [pc, #7]

@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
@ CHECK: error:
  ldrsh   pc, [r9, #10]
  ldrsh.w pc, [r10, #10]
  ldrsh   pc, [r11, #-5]
  ldrsh   pc, [pc, #7]
  ldrsh.w pc, [pc, #7]
