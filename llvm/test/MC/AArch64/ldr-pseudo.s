//RUN: llvm-mc  -triple=aarch64-linux-gnu -print-imm-hex %s | FileCheck %s

//
// Check that large constants are converted to ldr from constant pool
//
// simple test
.section a, "ax", @progbits
// CHECK-LABEL: f1:
f1:
  ldr x0, =0x1234
// CHECK: mov    x0, #0x1234
  ldr w1, =0x4567
// CHECK:  mov    w1, #0x4567
  ldr x0, =0x12340000
// CHECK:  mov    x0, #0x12340000
  ldr w1, =0x45670000
// CHECK: mov    w1, #0x45670000
  ldr x0, =0xabc00000000
// CHECK: mov    x0, #0xabc00000000
  ldr x0, =0xbeef000000000000
// CHECK: mov    x0, #-0x4111000000000000

.section b,"ax",@progbits
// CHECK-LABEL: f3:
f3:
  ldr w0, =0x10001
// CHECK: ldr w0, .Ltmp[[TMP0:[0-9]+]]

// loading multiple constants
.section c,"ax",@progbits
// CHECK-LABEL: f4:
f4:
  ldr w0, =0x10002
// CHECK: ldr w0, .Ltmp[[TMP1:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  ldr w0, =0x10003
// CHECK: ldr w0, .Ltmp[[TMP2:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1

// TODO: the same constants should have the same constant pool location
.section d,"ax",@progbits
// CHECK-LABEL: f5:
f5:
  ldr w0, =0x10004
// CHECK: ldr w0, .Ltmp[[TMP3:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  ldr w0, =0x10004
// CHECK: ldr w0, .Ltmp[[TMP3:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1

// a section defined in multiple pieces should be merged and use a single constant pool
.section e,"ax",@progbits
// CHECK-LABEL: f6:
f6:
  ldr w0, =0x10006
// CHECK: ldr w0, .Ltmp[[TMP5:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1

.section f, "ax", @progbits
// CHECK-LABEL: f7:
f7:
  adds x0, x0, #1
  adds x0, x0, #1
  adds x0, x0, #1

.section e, "ax", @progbits
// CHECK-LABEL: f8:
f8:
  adds x0, x0, #1
  ldr w0, =0x10007
// CHECK: ldr w0, .Ltmp[[TMP6:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1

//
// Check that symbols can be loaded using ldr pseudo
//

// load an undefined symbol
.section g,"ax",@progbits
// CHECK-LABEL: f9:
f9:
  ldr w0, =foo
// CHECK: ldr w0, .Ltmp[[TMP7:[0-9]+]]

// load a symbol from another section
.section h,"ax",@progbits
// CHECK-LABEL: f10:
f10:
  ldr w0, =f5
// CHECK: ldr w0, .Ltmp[[TMP8:[0-9]+]]

// load a symbol from the same section
.section i,"ax",@progbits
// CHECK-LABEL: f11:
f11:
  ldr w0, =f12
// CHECK: ldr w0, .Ltmp[[TMP9:[0-9]+]]
  ldr w0,=0x3C000
// CHECK: ldr     w0, .Ltmp[[TMP10:[0-9]+]]

// CHECK-LABEL: f12:
f12:
  adds x0, x0, #1
  adds x0, x0, #1

.section j,"ax",@progbits
// mix of symbols and constants
// CHECK-LABEL: f13:
f13:
  adds x0, x0, #1
  adds x0, x0, #1
  ldr w0, =0x101
// CHECK: mov w0, #0x101
  adds x0, x0, #1
  adds x0, x0, #1
  ldr w0, =bar
// CHECK: ldr w0, .Ltmp[[TMP11:[0-9]+]]
  adds x0, x0, #1
  adds x0, x0, #1
//
// Check for correct usage in other contexts
//

// usage in macro
.macro useit_in_a_macro
  ldr w0, =0x10008
  ldr w0, =baz
.endm
.section k,"ax",@progbits
// CHECK-LABEL: f14:
f14:
  useit_in_a_macro
// CHECK: ldr w0, .Ltmp[[TMP12:[0-9]+]]
// CHECK: ldr w0, .Ltmp[[TMP13:[0-9]+]]

// usage with expressions
.section l, "ax", @progbits
// CHECK-LABEL: f15:
f15:
  ldr w0, =0x10001+8
// CHECK: ldr w0, .Ltmp[[TMP14:[0-9]+]]
  adds x0, x0, #1
  ldr w0, =bar+4
// CHECK: ldr w0, .Ltmp[[TMP15:[0-9]+]]
  adds x0, x0, #1

// usage with 64-bit regs
.section m, "ax", @progbits
// CHECK-LABEL: f16:
f16:
  ldr x0, =0x0102030405060708
// CHECK: ldr x0, .Ltmp[[TMP16:[0-9]+]]
  add x0, x0, #1
  ldr w0, =bar
// CHECK: ldr w0, .Ltmp[[TMP17:[0-9]+]]
  ldr x0, =bar+16
// CHECK: ldr x0, .Ltmp[[TMP18:[0-9]+]]
  add x0, x0, #1
  ldr x0, =0x100000001
// CHECK: ldr x0, .Ltmp[[TMP19:[0-9]+]]
  ldr x1, =-0x80000001
// CHECK: ldr x1, .Ltmp[[TMP20:[0-9]+]]
  ldr x2, =0x10001
// CHECK: ldr x2, .Ltmp[[TMP21:[0-9]+]]

// check range for 32-bit regs
.section n, "ax", @progbits
// CHECK-LABEL: f17:
f17:
  ldr w0, =0xFFFFFFFF
// CHECK: ldr w0, .Ltmp[[TMP22:[0-9]+]]
  add w0, w0, #1
  ldr w1, =-0x7FFFFFFF
// CHECK: ldr w1, .Ltmp[[TMP23:[0-9]+]]
  add w0, w0, #1
  ldr w0, =-1
// CHECK: ldr w0, .Ltmp[[TMP24:[0-9]+]]
  add w0, w0, #1

// make sure the same contant uses different pools for 32- and 64-bit registers
.section o, "ax", @progbits
// CHECK-LABEL: f18:
f18:
  ldr w0, =0x320064
// CHECK: ldr w0, .Ltmp[[TMP25:[0-9]+]]
  add w0, w0, #1
  ldr x1, =0x320064
// CHECK: ldr x1, .Ltmp[[TMP26:[0-9]+]]

//
// Constant Pools
//
// CHECK: .section b,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP0]]
// CHECK: .word 65537

// CHECK: .section c,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP1]]
// CHECK: .word 65538
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP2]]
// CHECK: .word 65539

// CHECK: .section d,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP3]]
// CHECK: .word 65540

// CHECK: .section e,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP5]]
// CHECK: .word 65542
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP6]]
// CHECK: .word 65543

// Should not switch to section because it has no constant pool
// CHECK-NOT: .section f,"ax",@progbits

// CHECK: .section g,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP7]]
// CHECK: .word foo

// CHECK: .section h,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP8]]
// CHECK: .word f5

// CHECK: .section i,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP9]]
// CHECK: .word f12
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP10]]
// CHECK: .word 245760

// CHECK: .section j,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP11]]
// CHECK: .word bar

// CHECK: .section k,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP12]]
// CHECK: .word 65544
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP13]]
// CHECK: .word baz

// CHECK: .section l,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP14]]
// CHECK: .word 65545
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP15]]
// CHECK: .word bar+4

// CHECK: .section m,"ax",@progbits
// CHECK: .p2align 3
// CHECK: .Ltmp[[TMP16]]
// CHECK: .xword 72623859790382856
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP17]]
// CHECK: .word bar
// CHECK: .p2align 3
// CHECK: .Ltmp[[TMP18]]
// CHECK: .xword bar+16
// CHECK: .p2align 3
// CHECK: .Ltmp[[TMP19]]
// CHECK: .xword 4294967297
// CHECK: .p2align 3
// CHECK: .Ltmp[[TMP20]]
// CHECK: .xword -2147483649
// CHECK: .p2align 3
// CHECK: .Ltmp[[TMP21]]
// CHECK: .xword 65537

// CHECK: .section n,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP22]]
// CHECK: .word 4294967295
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP23]]
// CHECK: .word -2147483647
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP24]]
// CHECK: .word -1

// CHECK: .section o,"ax",@progbits
// CHECK: .p2align 2
// CHECK: .Ltmp[[TMP25]]
// CHECK: .word 3276900
