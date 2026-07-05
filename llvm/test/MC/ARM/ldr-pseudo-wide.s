@ Test case for PR30352
@ Check that ldr.w is:
@ accepted and ignored for ARM
@ accepted and propagated for Thumb2
@ rejected as needing Thumb2 for Thumb

@RUN: llvm-mc -triple armv5-unknown-linux-gnueabi %s | FileCheck --check-prefix=CHECK-ARM --check-prefix=CHECK %s
@RUN: llvm-mc -triple   armv7-base-apple-darwin %s | FileCheck --check-prefix=CHECK-DARWIN-ARM --check-prefix=CHECK-DARWIN %s
@RUN: llvm-mc -triple thumbv7-unknown-linux-gnueabi %s | FileCheck --check-prefix=CHECK-THUMB2 --check-prefix=CHECK %s
@RUN: llvm-mc -triple thumbv7-base-apple-darwin %s | FileCheck --check-prefix=CHECK-DARWIN-THUMB2 --check-prefix=CHECK-DARWIN %s
@RUN: not llvm-mc -triple thumbv6-unknown-linux-gnueabi %s 2>&1 | FileCheck --check-prefix=CHECK-THUMB %s
@RUN: not llvm-mc -triple thumbv6-base-apple-darwin %s 2>&1 | FileCheck --check-prefix=CHECK-THUMB %s
@ CHECK-LABEL: f1:
f1:
  ldr r0, =0x10002
@ CHECK-ARM: ldr r0, .Ltmp[[TMP0:[0-9]+]]
@ CHECK-DARWIN-ARM: ldr r0, Ltmp0
@ CHECK-THUMB2: ldr r0, .Ltmp[[TMP0:[0-9]+]]
@ CHECK-DARWIN-THUMB2: ldr r0, Ltmp0

  ldr.w r0, =0x10002
@ CHECK-ARM: ldr r0, .Ltmp[[TMP0]]
@ CHECK-DARWIN-ARM: ldr r0, Ltmp0
@ CHECK-THUMB2: ldr.w r0, .Ltmp[[TMP0:[0-9]+]]
@ CHECK-DARWIN-THUMB2: ldr.w r0, Ltmp0
@ CHECK-THUMB: error: instruction requires: thumb2
@ CHECK-THUMB-NEXT:  ldr.w r0, =0x10002

@ CHECK-LABEL: f2:
f2:
  ldr r0, =foo
@ CHECK-ARM: ldr r0, .Ltmp[[TMP1:[0-9]+]]
@ CHECK-DARWIN-ARM: ldr r0, Ltmp1
@ CHECK-THUMB2: ldr r0, .Ltmp[[TMP1:[0-9]+]]
@ CHECK-DARWIN-THUMB2: ldr r0, Ltmp1

  ldr.w r0, =foo
@ CHECK-ARM: ldr r0, .Ltmp[[TMP2:[0-9]+]]
@ CHECK-DARWIN-ARM: ldr r0, Ltmp1
@ CHECK-THUMB2: ldr.w r0, .Ltmp[[TMP2:[0-9]+]]
@ CHECK-DARWIN-THUMB2: ldr.w r0, Ltmp1
@ CHECK-THUMB: error: instruction requires: thumb2
@ CHECK-THUMB-NEXT:  ldr.w r0, =foo

@ CHECK-LABEL: f3:
f3:
  ldr.w r1, =0x1
@ CHECK-ARM: mov r1, #1
@ CHECK-DARWIN-ARM: mov r1, #1
@ CHECK-THUMB2: mov.w r1, #1
@ CHECK-DARWIN-THUMB2: mov.w r1, #1
@ CHECK-THUMB: error: instruction requires: thumb2
@ CHECK-THUMB-NEXT:  ldr.w r1, =0x1

@ CHECK: .Ltmp0:
@ CHECK-NEXT: .long   65538
@ CHECK: .Ltmp1:
@ CHECK-NEXT: .long   foo

@ CHECK-DARWIN: Ltmp0:
@ CHECK-DARWIN-NEXT: .long   65538
@ CHECK-DARWIN: Ltmp1:
@ CHECK-DARWIN-NEXT: .long   foo
