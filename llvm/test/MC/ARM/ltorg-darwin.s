@ This test has a partner (ltorg.s) that contains matching
@ tests for the .ltorg on linux targets. We need separate files
@ because the syntax for switching sections and temporary labels differs
@ between darwin and linux. Any tests added here should have a matching
@ test added there.

@RUN: llvm-mc -triple   armv7-apple-darwin %s | FileCheck %s
@RUN: llvm-mc -triple thumbv5-apple-darwin %s | FileCheck %s
@RUN: llvm-mc -triple thumbv7-apple-darwin %s | FileCheck %s

@ check that ltorg dumps the constant pool at the current location
.section __TEXT,a,regular,pure_instructions
@ CHECK-LABEL: f2:
f2:
  ldr r0, =0x10002
@ CHECK: ldr r0, Ltmp0
  adds r0, r0, #1
  adds r0, r0, #1
  b f3
.ltorg
@ constant pool
@ CHECK: .data_region
@ CHECK: .p2align 2
@ CHECK-LABEL: Ltmp0:
@ CHECK: .long 65538
@ CHECK: .end_data_region

@ CHECK-LABEL: f3:
f3:
  adds r0, r0, #1
  adds r0, r0, #1

@ check that ltorg clears the constant pool after dumping it
.section __TEXT,b,regular,pure_instructions
@ CHECK-LABEL: f4:
f4:
  ldr r0, =0x10003
@ CHECK: ldr r0, Ltmp1
  adds r0, r0, #1
  adds r0, r0, #1
  b f5
.ltorg
@ constant pool
@ CHECK: .data_region
@ CHECK: .p2align 2
@ CHECK-LABEL: Ltmp1:
@ CHECK: .long 65539
@ CHECK: .end_data_region

@ CHECK-LABEL: f5:
f5:
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =0x10004
@ CHECK: ldr r0, Ltmp2
  adds r0, r0, #1
  b f6
.ltorg
@ constant pool
@ CHECK: .data_region
@ CHECK: .p2align 2
@ CHECK-LABEL: Ltmp2:
@ CHECK: .long 65540
@ CHECK: .end_data_region

@ CHECK-LABEL: f6:
f6:
  adds r0, r0, #1
  adds r0, r0, #1

@ check that ltorg does not issue an error if there is no constant pool
.section __TEXT,c,regular,pure_instructions
@ CHECK-LABEL: f7:
f7:
  adds r0, r0, #1
  b f8
  .ltorg
f8:
  adds r0, r0, #1

@ check that ltorg works for labels
.section __TEXT,d,regular,pure_instructions
@ CHECK-LABEL: f9:
f9:
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =bar
@ CHECK: ldr r0, Ltmp3
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  b f10
.ltorg
@ constant pool
@ CHECK: .data_region
@ CHECK: .p2align 2
@ CHECK-LABEL: Ltmp3:
@ CHECK: .long bar
@ CHECK: .end_data_region

@ CHECK-LABEL: f10:
f10:
  adds r0, r0, #1
  adds r0, r0, #1

@ check that use of ltorg does not prevent dumping non-empty constant pools at end of section
.section __TEXT,e,regular,pure_instructions
@ CHECK-LABEL: f11:
f11:
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =0x10005
@ CHECK: ldr r0, Ltmp4
  b f12
  .ltorg
@ constant pool
@ CHECK: .data_region
@ CHECK: .p2align 2
@ CHECK-LABEL: Ltmp4:
@ CHECK: .long 65541
@ CHECK: .end_data_region

@ CHECK-LABEL: f12:
f12:
  adds r0, r0, #1
  ldr r0, =0x10006
@ CHECK: ldr r0, Ltmp5

.section __TEXT,f,regular,pure_instructions
@ CHECK-LABEL: f13
f13:
  adds r0, r0, #1
  adds r0, r0, #1

@ should not have a constant pool at end of section with empty constant pools
@ CHECK-NOT: .section __TEXT,a,regular,pure_instructions
@ CHECK-NOT: .section __TEXT,b,regular,pure_instructions
@ CHECK-NOT: .section __TEXT,c,regular,pure_instructions
@ CHECK-NOT: .section __TEXT,d,regular,pure_instructions

@ should have a non-empty constant pool at end of this section
@ CHECK: .section __TEXT,e,regular,pure_instructions
@ constant pool
@ CHECK: .data_region
@ CHECK: .p2align 2
@ CHECK-LABEL: Ltmp5:
@ CHECK: .long 65542
@ CHECK: .end_data_region

@ should not have a constant pool at end of section with empty constant pools
@ CHECK-NOT: .section __TEXT,f,regular,pure_instructions
