@RUN: llvm-mc -triple armv7-unknown-linux-gnueabi %s | FileCheck --check-prefix=CHECK-ARM --check-prefix=CHECK %s
@RUN: llvm-mc -triple thumbv7-unknown-linux-gnueabi %s | FileCheck --check-prefix=CHECK-THUMB2 --check-prefix=CHECK %s

@
@ Check that ldr to constant pool correctly transfers the condition codes
@
@ simple test
.section a,"ax",%progbits
@ CHECK-LABEL: f0:
f0:
  it eq
  ldreq r0, =0x10002
@ CHECK: ldreq r0, .Ltmp[[TMP0:[0-9]+]]

@ loading multiple constants
.section b,"ax",%progbits
@ CHECK-LABEL: f1:
f1:
  ite eq
  ldreq r0, =0x10003
@ CHECK: ldreq r0, .Ltmp[[TMP1:[0-9]+]]
  ldrne r0, =0x10004
@ CHECK: ldrne r0, .Ltmp[[TMP2:[0-9]+]]

@ transformation to mov
.section c, "ax", %progbits
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
@ CHECK movwne r2, #4660

@
@ Constant Pools
@
@ CHECK: .section a,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP0]]
@ CHECK: .long 65538

@ CHECK: .section b,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP1]]
@ CHECK: .long 65539
@ CHECK: .Ltmp[[TMP2]]
@ CHECK: .long 65540
