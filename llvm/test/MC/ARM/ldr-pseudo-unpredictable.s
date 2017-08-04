@RUN: llvm-mc -triple armv5-unknown-linux-gnueabi %s | FileCheck --check-prefix=CHECK-ARM %s
@RUN: llvm-mc -triple thumbv7-unknown-linux-gnueabi %s 2>&1 | FileCheck --check-prefix=CHECK-T2 %s
@RUN: not llvm-mc -triple thumbv5-unknown-linux-gnueabi %s 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
@RUN: llvm-mc -triple armv5-base-apple-darwin %s | FileCheck --check-prefix=CHECK-DARWIN-ARM %s
@RUN: llvm-mc -triple thumbv7-base-apple-darwin %s 2>&1 | FileCheck --check-prefix=CHECK-DARWIN-T2 %s
@RUN: not llvm-mc -triple thumbv5-base.apple.darwin %s 2>&1 | FileCheck --check-prefix=CHECK-NONE %s

@ We dont't do the transformation for rt = sp or pc
@ as it is unpredictable for many of the MOV encondings
  ldr pc, = 0x4
@ CHECK-ARM: ldr pc, .Ltmp[[TMP0:[0-9]+]]
@ CHECK-DARWIN-ARM: ldr pc, Ltmp0
@ CHECK-T2: ldr.w pc, .Ltmp[[TMP0:[0-9]+]]
@ CHECK-DARWIN-T2: ldr.w pc, Ltmp0
@ CHECK-NONE: error: instruction requires: thumb2
  ldr sp, = 0x8
@ CHECK-ARM: ldr sp, .Ltmp[[TMP1:[0-9]+]]
@ CHECK-DARWIN-ARM: ldr sp, Ltmp1
@ CHECK-T2: ldr.w sp, .Ltmp[[TMP1:[0-9]+]]
@ CHECK-DARWIN-T2: ldr.w sp, Ltmp1
@ CHECK-NONE: error: instruction requires: thumb2
