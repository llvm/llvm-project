@ PR28647
@ RUN: not llvm-mc -triple=thumbv7a-linux-gnueabi -filetype=obj < %s 2>&1 | FileCheck %s
    .text
    .syntax unified
    .balign 2

@ mov with :upper16: or :lower16: should not match mov with modified immediate
    mov r0, :upper16: sym0
@ CHECK: error: instruction requires: arm-mode
    mov r0, :lower16: sym0
@ CHECK: error: instruction requires: arm-mode
    .equ sym0, 0x01abcdef
