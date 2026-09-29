# RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s -o %t
# RUN: llvm-readelf -Ss %t | FileCheck %s

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

# CHECK: [[#TEXT:]]] .text

# CHECK:      1: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT [[#TEXT]] $x
# CHECK-NEXT: 2: 0000000000000004     0 NOTYPE  LOCAL  DEFAULT [[#TEXT]] $d
# CHECK-NEXT: 3: 0000000000000064     0 NOTYPE  LOCAL  DEFAULT [[#TEXT]] $x
# CHECK-NEXT: 4: 0000000000000068     0 NOTYPE  LOCAL  DEFAULT [[#TEXT]] $x
# CHECK-NEXT: 5: 0000000000000068     0 NOTYPE  GLOBAL DEFAULT [[#TEXT]] $d
# CHECK-NOT:  {{.}}
