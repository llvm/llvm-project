@ This test has a partner (ldr-pseudo-darwin.s) that contains matching
@ tests for the ldr-pseudo on darwin targets. We need separate files
@ because the syntax for switching sections and temporary labels differs
@ between darwin and linux. Any tests added here should have a matching
@ test added there.

@RUN: llvm-mc -triple   armv7-unknown-linux-gnueabi %s | FileCheck %s
@RUN: llvm-mc -triple thumbv5-unknown-linux-gnueabi %s | FileCheck %s
@RUN: llvm-mc -triple thumbv7-unknown-linux-gnueabi %s | FileCheck %s

@
@ Check that large constants are converted to ldr from constant pool
@
@ simple test
.section b,"ax",%progbits
@ CHECK-LABEL: f3:
f3:
  ldr r0, =0x10001
@ CHECK: ldr r0, .Ltmp[[TMP0:[0-9]+]]

@ loading multiple constants
.section c,"ax",%progbits
@ CHECK-LABEL: f4:
f4:
  ldr r0, =0x10002
@ CHECK: ldr r0, .Ltmp[[TMP1:[0-9]+]]
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =0x10003
@ CHECK: ldr r0, .Ltmp[[TMP2:[0-9]+]]
  adds r0, r0, #1
  adds r0, r0, #1

@ TODO: the same constants should have the same constant pool location
.section d,"ax",%progbits
@ CHECK-LABEL: f5:
f5:
  ldr r0, =0x10004
@ CHECK: ldr r0, .Ltmp[[TMP3:[0-9]+]]
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  adds r0, r0, #1
  ldr r0, =0x10004
@ CHECK: ldr r0, .Ltmp[[TMP4:[0-9]+]]
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
  ldr r0, =0x101
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
  ldr r0, =0x10008
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
  ldr r0, =0x10001+8
@ CHECK: ldr r0, .Ltmp[[TMP14:[0-9]+]]
  adds r0, r0, #1
  ldr r0, =bar+4
@ CHECK: ldr r0, .Ltmp[[TMP15:[0-9]+]]
  adds r0, r0, #1

@
@ Constant Pools
@
@ CHECK: .section b,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP0]]
@ CHECK: .long 65537

@ CHECK: .section c,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP1]]
@ CHECK: .long 65538
@ CHECK: .Ltmp[[TMP2]]
@ CHECK: .long 65539

@ CHECK: .section d,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP3]]
@ CHECK: .long 65540
@ CHECK: .Ltmp[[TMP4]]
@ CHECK: .long 65540

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
@ CHECK: .long 257
@ CHECK: .Ltmp[[TMP11]]
@ CHECK: .long bar

@ CHECK: .section k,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP12]]
@ CHECK: .long 65544
@ CHECK: .Ltmp[[TMP13]]
@ CHECK: .long baz

@ CHECK: .section l,"ax",%progbits
@ CHECK: .p2align 2
@ CHECK: .Ltmp[[TMP14]]
@ CHECK: .long 65545
@ CHECK: .Ltmp[[TMP15]]
@ CHECK: .long bar+4
