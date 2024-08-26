// RUN: llvm-mc -triple=aarch64 -filetype=obj %s | llvm-objdump -t - | FileCheck %s --match-full-lines
// RUN: llvm-mc -triple=aarch64 -filetype=obj -implicit-mapsyms %s | llvm-objdump -t - | FileCheck %s --check-prefix=CHECK1 --match-full-lines

/// The test covers many state transitions. Let's use the first state and the last state to describe a section.
/// .text goes through cd -> dd -> cc -> dd.
/// .data goes through dd -> dc -> cd.
.file "0.s"
.section .text1,"ax"
add w0, w0, w0

.text
add w0, w0, w0
.word 42

.pushsection .data,"aw"
.word 42
.popsection

.text
.word 42

.section .text1,"ax"
add w1, w1, w1

.text
add w1, w1, w1

.section .data,"aw"
.word 42
add w0, w0, w0

.text
.word 42

## .rodata and subsequent symbols should be after the FILE symbol of "1.s".
.file "1.s"
.section .rodata,"a"
.word 42
add w0, w0, w0

.section .data,"aw"
add w0, w0, w0
.word 42

.text

.ident "clang"
.section ".note.GNU-stack","",@progbits

// CHECK:      SYMBOL TABLE:
// CHECK-NEXT: 0000000000000000 l    df *ABS*	0000000000000000 0.s
// CHECK-NEXT: 0000000000000000 l       .text1	0000000000000000 $x
// CHECK-NEXT: 0000000000000000 l       .text	0000000000000000 $x
// CHECK-NEXT: 0000000000000004 l       .text	0000000000000000 $d
// CHECK-NEXT: 0000000000000000 l       .data	0000000000000000 $d
// CHECK-NEXT: 000000000000000c l       .text	0000000000000000 $x
// CHECK-NEXT: 0000000000000008 l       .data	0000000000000000 $x
// CHECK-NEXT: 0000000000000010 l       .text	0000000000000000 $d
// CHECK-NEXT: 0000000000000000 l    df *ABS*	0000000000000000 1.s
// CHECK-NEXT: 0000000000000000 l       .rodata	0000000000000000 $d
// CHECK-NEXT: 0000000000000004 l       .rodata	0000000000000000 $x
// CHECK-NEXT: 0000000000000010 l       .data	0000000000000000 $d
// CHECK-NEXT: 0000000000000000 l       .comment	0000000000000000 $d
// CHECK-NOT:  {{.}}

// CHECK1:      SYMBOL TABLE:
// CHECK1-NEXT: 0000000000000000 l    df *ABS*	0000000000000000 0.s
// CHECK1-NEXT: 0000000000000004 l       .text	0000000000000000 $d
// CHECK1-NEXT: 000000000000000c l       .text	0000000000000000 $x
// CHECK1-NEXT: 0000000000000008 l       .data	0000000000000000 $x
// CHECK1-NEXT: 0000000000000010 l       .text	0000000000000000 $d
// CHECK1-NEXT: 0000000000000014 l       .text	0000000000000000 $x
// CHECK1-NEXT: 0000000000000000 l    df *ABS*	0000000000000000 1.s
// CHECK1-NEXT: 0000000000000004 l       .rodata	0000000000000000 $x
// CHECK1-NEXT: 0000000000000008 l       .rodata	0000000000000000 $d
// CHECK1-NEXT: 0000000000000010 l       .data	0000000000000000 $d
// CHECK1-NOT:  {{.}}
