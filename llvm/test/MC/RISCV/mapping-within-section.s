# RUN: llvm-mc -triple=riscv32 -filetype=obj %s | llvm-readelf -Ss - | FileCheck %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj %s | llvm-readelf -Ss - | FileCheck %s

        .text
# $x at 0x0000
        nop
# $d at 0x0004
        .ascii "012"
        .byte 1
        .hword 2
        .word 4
        .single 4.0
        .double 8.0
        .space 10
        .zero 3
        .fill 10, 2, 42
        .org 100, 12
# $x at 0x0064
        nop

## Capture section index.
# CHECK: [[#TEXT:]]] .text

# CHECK:    Value  Size Type    Bind   Vis     Ndx       Name
# CHECK: 00000000     0 NOTYPE  LOCAL  DEFAULT [[#TEXT]] $x
# CHECK: 00000004     0 NOTYPE  LOCAL  DEFAULT [[#TEXT]] $d
# CHECK: 00000064     0 NOTYPE  LOCAL  DEFAULT [[#TEXT]] $x
