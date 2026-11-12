@ This test has a partner (ldr-pseudo-darwin.s) that contains matching
@ tests for the ldr-pseudo on darwin targets. We need separate files
@ because the syntax for switching sections and temporary labels differs
@ between darwin and linux. Any tests added here should have a matching
@ test added there.

@RUN: llvm-mc -triple   armv7-unknown-linux-gnueabi %s | FileCheck --check-prefix=CHECK-ARM --check-prefix=CHECK %s
@RUN: llvm-mc -triple   armv5-unknown-linux-gnueabi %s | FileCheck --check-prefix=CHECK-ARMV5 --check-prefix=CHECK %s
@RUN: llvm-mc -triple thumbv5-unknown-linux-gnueabi %s | FileCheck --check-prefix=CHECK-THUMB --check-prefix=CHECK %s
@RUN: llvm-mc -triple thumbv7-unknown-linux-gnueabi %s | FileCheck --check-prefix=CHECK-THUMB2 --check-prefix=CHECK %s
@RUN: llvm-mc -triple thumbv8m.base-unknown-linux-gnueabi %s | FileCheck --check-prefix=CHECK-BASELINE --check-prefix=CHECK %s
@
@ Check that large constants are converted to ldr from constant pool
@
@ simple test
.section b,"ax",%progbits
@ CHECK-LABEL: f3:
f3:
  ldr r0, =0x10002
@ CHECK: ldr r0, .Ltmp[[TMP0:[0-9]+]]

@ loading multiple constants
.section c,"ax",%progbits
@ CHECK-LABEL: f4:
f4:
  ldr r0, =0x10003
@ CHECK: ldr r0, .Ltmp[[TMP1:[0-9]+]]
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =0x10004
@ CHECK: ldr r0, .Ltmp[[TMP2:[0-9]+]]
  adds r0, r0, #1
  adds r0, r0, #1

@ TODO: the same constants should have the same constant pool location
.section d,"ax",%progbits
@ CHECK-LABEL: f5:
f5:
  ldr r0, =0x10005
@ CHECK: ldr r0, .Ltmp[[TMP3:[0-9]+]]
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =0x10005
@ CHECK: ldr r0, .Ltmp[[TMP3:[0-9]+]]
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1

@ a section defined in multiple pieces should be merged and use a single constant pool
.section e,"ax",%progbits
@ CHECK-LABEL: f6:
f6:
  ldr r0, =0x10006
@ CHECK: ldr r0, .Ltmp[[TMP5:[0-9]+]]
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1

.section f, "ax", %progbits
@ CHECK-LABEL: f7:
f7:
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1

.section e, "ax", %progbits
@ CHECK-LABEL: f8:
f8:
  adds r0, r0, #1
  ldr r0, =0x10007
@ CHECK: ldr r0, .Ltmp[[TMP6:[0-9]+]]
  adds r0, r0, #1
  adds r0, r0, #1

@
@ Check that symbols can be loaded using ldr pseudo
@

@ load an undefined symbol
.section g,"ax",%progbits
@ CHECK-LABEL: f9:
f9:
  ldr r0, =foo
@ CHECK: ldr r0, .Ltmp[[TMP7:[0-9]+]]

@ load a symbol from another section
.section h,"ax",%progbits
@ CHECK-LABEL: f10:
f10:
  ldr r0, =f5
@ CHECK: ldr r0, .Ltmp[[TMP8:[0-9]+]]

@ load a symbol from the same section
.section i,"ax",%progbits
@ CHECK-LABEL: f11:
f11:
  ldr r0, =f12
@ CHECK: ldr r0, .Ltmp[[TMP9:[0-9]+]]

@ CHECK-LABEL: f12:
f12:
  adds r0, r0, #1
  adds r0, r0, #1

.section j,"ax",%progbits
@ mix of symbols and constants
@ CHECK-LABEL: f13:
f13:
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =0x10008
@ CHECK: ldr r0, .Ltmp[[TMP10:[0-9]+]]
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =bar
@ CHECK: ldr r0, .Ltmp[[TMP11:[0-9]+]]
  adds r0, r0, #1
  adds r0, r0, #1
@
@ Check for correct usage in other contexts
@

@ usage in macro
.macro useit_in_a_macro
  ldr r0, =0x10009
  ldr r0, =baz
.endm
.section k,"ax",%progbits
@ CHECK-LABEL: f14:
f14:
  useit_in_a_macro
@ CHECK: ldr r0, .Ltmp[[TMP12:[0-9]+]]
@ CHECK: ldr r0, .Ltmp[[TMP13:[0-9]+]]

@ usage with expressions
.section l, "ax", %progbits
@ CHECK-LABEL: f15:
f15:
  ldr r0, =0x10001+9
@ CHECK: ldr r0, .Ltmp[[TMP14:[0-9]+]]
  adds r0, r0, #1
  ldr r0, =bar+4
@ CHECK: ldr r0, .Ltmp[[TMP15:[0-9]+]]
  adds r0, r0, #1

@ transformation to mov
.section m, "ax", %progbits
@ CHECK-LABEL: f16:
f16:

@ Representable in ARM, and Thumb with support mov.w or movw
  ldr r1, =0x1
@ CHECK-ARM: mov r1, #1
@ CHECK-ARMV5: mov r1, #1
@ CHECK-THUMB: ldr r1, .Ltmp[[TMP16:[0-9]+]]
@ CHECK-THUMB2: mov.w r1, #1
@ CHECK-BASELINE: movw r1, #1

@ Immediate is representable in A1 and T2 modified immediate only not movw
  ldr r2, =0x120000
@ CHECK-ARM: mov r2, #1179648
@ CHECK-ARMV5: mov r2, #1179648
@ CHECK-THUMB: ldr r2, .Ltmp[[TMP17:[0-9]+]]
@ CHECK-THUMB2: mov.w r2, #1179648
@ CHECK-BASELINE: ldr r2, .Ltmp[[TMP16:[0-9]+]]

@ Immediate can be represented only with movw instruction
  ldr r3, =0x1234
@ CHECK-ARM: movw r3, #4660
@ CHECK-ARMV5: ldr r3, .Ltmp[[TMP16:[0-9]+]]
@ CHECK-THUMB: ldr r3, .Ltmp[[TMP18:[0-9]+]]
@ CHECK-THUMB2: movw r3, #4660
@ CHECK-BASELINE: movw r3, #4660

@ Immediate can be represented only with T2 modified immediate
  ldr r4, =0xabababab
@ CHECK-ARM: ldr r4, .Ltmp[[TMP16:[0-9]+]]
@ CHECK-ARMV5: ldr r4, .Ltmp[[TMP17:[0-9]+]]
@ CHECK-THUMB: ldr r4, .Ltmp[[TMP19:[0-9]+]]
@ CHECK-THUMB2: mov.w r4, #2880154539
@ CHECK-BASELINE: ldr r4, .Ltmp[[TMP17:[0-9]+]]

@ Immediate can be represented only with A1 modified immediate
  ldr r5, =0x1000000b
@ CHECK-ARM: mov r5, #268435467
@ CHECK-ARMV5: mov r5, #268435467
@ CHECK-THUMB: ldr r5, .Ltmp[[TMP20:[0-9]+]]
@ CHECK-THUMB2: ldr r5, .Ltmp[[TMP16:[0-9]+]]
@ CHECK-BASELINE: ldr r5, .Ltmp[[TMP18:[0-9]+]]

@ Negative numbers can be used with MVN or in Thumb2 with modified immediate
  ldr r6, =-1
@ CHECK-ARM: mvn r6, #0
@ CHECK-ARMV5: mvn r6, #0
@ CHECK-THUMB: ldr r6, .Ltmp[[TMP21:[0-9]+]]
@ CHECK-THUMB2: mov.w r6, #-1
@ CHECK-BASELINE: ldr r6, .Ltmp[[TMP19:[0-9]+]]
  ldr r7, =-0x100
@ CHECK-ARM: mvn r7, #255
@ CHECK-ARMV5: mvn r7, #255
@ CHECK-THUMB: ldr r7, .Ltmp[[TMP22:[0-9]+]]
@ CHECK-THUMB2: mvn r7, #255
@ CHECK-BASELINE: ldr r7, .Ltmp[[TMP20:[0-9]+]]

@ Constant expressions can be used
  .equ expr, 0x10 + 0x10
  ldr r0, = expr
@ CHECK-ARM: mov r0, #32
@ CHECK-ARMV5: mov r0, #32
@ CHECK-THUMB: ldr r0, .Ltmp[[TMP23:[0-9]+]]
@ CHECK-THUMB2: mov.w r0, #32
@ CHECK-BASELINE: movw r0, #32
  ldr r1, = expr - 0x10
@ CHECK-ARM: mov r1, #16
@ CHECK-ARMV5: mov r1, #16
@ CHECK-THUMB: ldr r1, .Ltmp[[TMP24:[0-9]+]]
@ CHECK-THUMB2: mov.w r1, #16
@ CHECK-BASELINE: movw r1, #16

@ usage of translation in macro
.macro usemov_in_a_macro
  ldr r2, =0x3
  ldr r3, =expr
.endm
@ CHECK-LABEL: f17:
f17:
  usemov_in_a_macro
@ CHECK-ARM: mov r2, #3
@ CHECK-ARM: mov r3, #32
@ CHECK-ARMV5: mov r2, #3
@ CHECK-ARMV5: mov r3, #32
@ CHECK-THUMB: ldr r2, .Ltmp[[TMP25:[0-9]+]]
@ CHECK-THUMB: ldr r3, .Ltmp[[TMP26:[0-9]+]]
@ CHECK-THUMB2: mov.w r2, #3
@ CHECK-THUMB2: mov.w r3, #32
@ CHECK-BASELINE: movw r2, #3
@ CHECK-BASELINE: movw r3, #32
@
@ Constant Pools
@
@ CHECK: .section b,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP0]]
@ CHECK: .long 65538

@ CHECK: .section c,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP1]]
@ CHECK: .long 65539
@ CHECK: .Ltmp[[TMP2]]
@ CHECK: .long 65540

@ CHECK: .section d,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP3]]
@ CHECK: .long 65541

@ CHECK: .section e,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP5]]
@ CHECK: .long 65542
@ CHECK: .Ltmp[[TMP6]]
@ CHECK: .long 65543

@ Should not switch to section because it has no constant pool
@ CHECK-NOT: .section f,"ax",%progbits

@ CHECK: .section g,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP7]]
@ CHECK: .long foo

@ CHECK: .section h,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP8]]
@ CHECK: .long f5

@ CHECK: .section i,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP9]]
@ CHECK: .long f12

@ CHECK: .section j,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP10]]
@ CHECK: .long 65544
@ CHECK: .Ltmp[[TMP11]]
@ CHECK: .long bar

@ CHECK: .section k,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP12]]
@ CHECK: .long 65545
@ CHECK: .Ltmp[[TMP13]]
@ CHECK: .long baz

@ CHECK: .section l,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP14]]
@ CHECK: .long 65546
@ CHECK: .Ltmp[[TMP15]]
@ CHECK: .long bar+4
