// RUN: llvm-mc -triple=aarch64 -filetype=obj %s | llvm-objdump -t - | FileCheck %s

.section .text1,"ax"
add w0, w0, w0

.text
add w0, w0, w0
.word 42

.pushsection .data,"aw"
.word 42
.popsection

.text
add w1, w1, w1

.section .text1,"ax"
add w1, w1, w1

.text
.word 42

.section .rodata,"a"
.word 42
add w0, w0, w0

.ident "clang"
.section ".note.GNU-stack","",@progbits

// CHECK:      SYMBOL TABLE:
// CHECK-NEXT: 0000000000000000 l       .text1 0000000000000000 $x.0
// CHECK-NEXT: 0000000000000000 l       .text  0000000000000000 $x.1
// CHECK-NEXT: 0000000000000004 l       .text  0000000000000000 $d.2
// CHECK-NEXT: 0000000000000000 l       .data  0000000000000000 $d.3
// CHECK-NEXT: 0000000000000008 l       .text  0000000000000000 $x.4
// CHECK-NEXT: 000000000000000c l       .text  0000000000000000 $d.5
// CHECK-NEXT: 0000000000000000 l       .rodata        0000000000000000 $d.6
// CHECK-NEXT: 0000000000000004 l       .rodata        0000000000000000 $x.7
// CHECK-NEXT: 0000000000000000 l       .comment       0000000000000000 $d.8
// CHECK-NOT:  {{.}}
