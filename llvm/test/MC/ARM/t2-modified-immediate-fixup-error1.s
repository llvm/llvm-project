@ PR28647
@ RUN: not llvm-mc -triple=thumbv7a-linux-gnueabi -filetype=obj < %s 2>&1 | FileCheck %s
    .text
    .syntax unified
    .balign 2

@ Error with unencodeable immediate
    add r1, r2, sym0
@ CHECK: error: out of range immediate fixup value
    .equ sym0, 0x01abcdef
.L2:
    mov r0, .L2
@ CHECK: :[[#@LINE-1]]:5: error: unsupported relocation type
