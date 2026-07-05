@ This test has a partner (ltorg-darwin.s) that contains matching
@ tests for the .ltorg on darwin targets. We need separate files
@ because the syntax for switching sections and temporary labels differs
@ between darwin and linux. Any tests added here should have a matching
@ test added there.

@RUN: llvm-mc -triple   armv7-unknown-linux-gnueabi %s | FileCheck %s
@RUN: llvm-mc -triple thumbv5-unknown-linux-gnueabi %s | FileCheck %s
@RUN: llvm-mc -triple thumbv7-unknown-linux-gnueabi %s | FileCheck %s

@ check that ltorg dumps the constant pool at the current location
.section a,"ax",%progbits
@ CHECK-LABEL: f2:
f2:
  ldr r0, =0x10002
@ CHECK: ldr r0, .Ltmp[[TMP0:[0-9+]]]
  adds r0, r0, #1
  adds r0, r0, #1
  b f3
.ltorg
@ constant pool
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP0]]
@ CHECK: .long 65538

@ CHECK-LABEL: f3:
f3:
  adds r0, r0, #1
  adds r0, r0, #1

@ check that ltorg clears the constant pool after dumping it
.section b,"ax",%progbits
@ CHECK-LABEL: f4:
f4:
  ldr r0, =0x10003
@ CHECK: ldr r0, .Ltmp[[TMP1:[0-9+]]]
  adds r0, r0, #1
  adds r0, r0, #1
  b f5
.ltorg
@ constant pool
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP1]]
@ CHECK: .long 65539

@ CHECK-LABEL: f5:
f5:
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =0x10004
@ CHECK: ldr r0, .Ltmp[[TMP2:[0-9+]]]
  adds r0, r0, #1
  b f6
.ltorg
@ constant pool
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP2]]
@ CHECK: .long 65540

@ CHECK-LABEL: f6:
f6:
  adds r0, r0, #1
  adds r0, r0, #1

@ check that ltorg does not issue an error if there is no constant pool
.section c,"ax",%progbits
@ CHECK-LABEL: f7:
f7:
  adds r0, r0, #1
  b f8
  .ltorg
f8:
  adds r0, r0, #1

@ check that ltorg works for labels
.section d,"ax",%progbits
@ CHECK-LABEL: f9:
f9:
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =bar
@ CHECK: ldr r0, .Ltmp[[TMP3:[0-9+]]]
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  b f10
.ltorg
@ constant pool
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP3]]
@ CHECK: .long bar

@ CHECK-LABEL: f10:
f10:
  adds r0, r0, #1
  adds r0, r0, #1

@ check that use of ltorg does not prevent dumping non-empty constant pools at end of section
.section e,"ax",%progbits
@ CHECK-LABEL: f11:
f11:
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =0x10005
@ CHECK: ldr r0, .Ltmp[[TMP4:[0-9+]]]
  b f12
  .ltorg
@ constant pool
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP4]]
@ CHECK: .long 65541
@ CHECK-LABEL: f12:
f12:
  adds r0, r0, #1
  ldr r0, =0x10006
@ CHECK: ldr r0, .Ltmp[[TMP5:[0-9+]]]

.section f,"ax",%progbits
@ CHECK-LABEL: f13
f13:
  adds r0, r0, #1
  adds r0, r0, #1

@ should not have a constant pool at end of section with empty constant pools
@ CHECK-NOT: .section a,"ax",%progbits
@ CHECK-NOT: .section b,"ax",%progbits
@ CHECK-NOT: .section c,"ax",%progbits
@ CHECK-NOT: .section d,"ax",%progbits

@ should have a non-empty constant pool at end of this section
@ CHECK: .section e,"ax",%progbits
@ constant pool
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP5]]
@ CHECK: .long 65542

@ should not have a constant pool at end of section with empty constant pools
@ CHECK-NOT: .section f,"ax",%progbits
