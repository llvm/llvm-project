// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s | llvm-nm --no-sort --special-syms - | FileCheck %s --match-full-lines

    .text
// $x at 0x0000
    add w0, w0, w0
// $d at 0x0004
    .ascii "012"
    .byte 1
    .hword 2
    .word 4
    .xword 8
    .single 4.0
    .double 8.0
    .space 10
    .zero 3
    .fill 10, 2, 42
    .org 100, 12
// $x at 0x0018
    add x0, x0, x0

.globl $d
$d:
$x:

// CHECK:      0000000000000000 t $x
// CHECK-NEXT: 0000000000000004 t $d
// CHECK-NEXT: 0000000000000064 t $x
// CHECK-NEXT: 0000000000000068 t $x
// CHECK-NEXT: 0000000000000068 T $d
// CHECK-NOT:  {{.}}
