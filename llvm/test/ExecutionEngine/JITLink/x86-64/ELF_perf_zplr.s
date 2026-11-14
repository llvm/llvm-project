# REQUIRES: native && x86_64-linux
# UNSUPPORTED: wsl1, msan
#
# RUN: llvm-mc -triple=x86_64-linux -filetype=obj %s -o %t.o
# RUN: %python %S/../Inputs/check-perf-jitdump.py --run \
# RUN:   llvm-jitlink -perf-support %t.o | FileCheck %s
# CHECK: main: valid per-function unwind record
#
# The personality and LSDA belong to the executor's exception handler. They
# must not be copied as stale pointers into perf's independently loaded ELF.
# This input has zPLR, an indirect PC-relative personality, and a PC-relative
# LSDA. The checker requires a canonical zR CIE and no FDE LSDA augmentation.

        .text
        .globl main
        .type main,@function
main:
        .cfi_startproc
        .cfi_personality 0x9b, .Lpersonality
        .cfi_lsda 0x1b, .Llsda
        xorl %eax, %eax
        retq
        .cfi_endproc
        .size main, .-main

        .data
        .p2align 3
.Lpersonality:
        .quad 0
        .section .gcc_except_table,"a",@progbits
.Llsda:
        .byte 0xff, 0xff, 1, 0
        .section .note.GNU-stack,"",@progbits
