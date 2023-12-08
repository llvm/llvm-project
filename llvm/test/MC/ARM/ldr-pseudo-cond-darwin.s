@RUN: llvm-mc -triple armv7-base-apple-darwin %s | FileCheck --check-prefix=CHECK %s
@RUN: llvm-mc -triple thumbv7-base-apple-darwin %s | FileCheck --check-prefix=CHECK %s

@
@ Check that ldr to constant pool correctly transfers the condition codes
@
@ simple test
.section __TEXT,a,regular,pure_instructions
@ CHECK-LABEL: f0:
f0:
  it eq
  ldreq r0, =0x10002
@ CHECK: ldreq r0, Ltmp0

@ loading multiple constants
.section __TEXT,b,regular,pure_instructions
@ CHECK-LABEL: f1:
f1:
  ite eq
  ldreq r0, =0x10003
@ CHECK: ldreq r0, Ltmp1
  ldrne r0, =0x10004
@ CHECK: ldrne r0, Ltmp2

@ transformation to mov
.section __TEXT,d,regular,pure_instructions
@ CHECK-LABEL: f2:
f2:
@ Can use the narrow Thumb mov as it does not set flags in an IT block
  it eq
  ldreq r1, =0x1
@ CHECK: moveq r1, #1

@ Must use the wide Thumb mov if the constant can't be represented
  ite eq
  ldreq r2, = 0x1f000000
@ CHECK-ARM moveq r2, #520093696
@ CHECK-THUMB2 moveq.w r2, #520093696
  ldrne r3, = 0x00001234
@ CHECK: movwne r3, #4660

@
@ Constant Pools
@
@ CHECK: .section __TEXT,a,regular,pure_instructions
@ CHECK: .p2align 2
@ CHECK: Ltmp0:
@ CHECK: .long 65538

@ CHECK: .section __TEXT,b,regular,pure_instructions
@ CHECK: .p2align 2
@ CHECK: Ltmp1:
@ CHECK: .long 65539
@ CHECK: Ltmp2:
@ CHECK: .long 65540
