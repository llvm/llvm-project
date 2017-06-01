@RUN: llvm-mc -triple armv5-unknown-linux-gnueabi %s | FileCheck --check-prefix=CHECK-ARM %s
@RUN: not llvm-mc -triple thumbv7-unknown-linux-gnueabi %s 2>&1 | FileCheck --check-prefix=CHECK-SP %s
@RUN: not llvm-mc -triple thumbv5-unknown-linux-gnueabi %s 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
@RUN: llvm-mc -triple armv5-base-apple-darwin %s | FileCheck --check-prefix=CHECK-DARWIN-ARM %s
@RUN: not llvm-mc -triple thumbv7-base-apple-darwin %s 2>&1 | FileCheck --check-prefix=CHECK-DARWIN-SP %s
@RUN: not llvm-mc -triple thumbv5-base.apple.darwin %s 2>&1 | FileCheck --check-prefix=CHECK-NONE %s

@ We dont't do the transformation for rt = sp or pc
@ as it is unpredictable for many of the MOV encondings
  ldr pc, = 0x4
@ CHECK-ARM: ldr pc, .Ltmp[[TMP0:[0-9]+]]
@ CHECK-DARWIN-ARM: ldr pc, Ltmp0
@ CHECK-SP: error: instruction requires: arm-mode
@ CHECK-DARWIN-SP: error: instruction requires: arm-mode
@ CHECK-NONE: error: instruction requires: arm-mode
  ldr sp, = 0x8
@ CHECK-ARM: ldr sp, .Ltmp[[TMP1:[0-9]+]]
@ CHECK-DARWIN-ARM: ldr sp, Ltmp1
@ CHECK-SP: ldr.w sp, .Ltmp[[TMP0:[0-9]+]]
@ CHECK-DARWIN-SP: ldr.w sp, Ltmp0
@ CHECK-NONE: error: instruction requires: arm-mode
