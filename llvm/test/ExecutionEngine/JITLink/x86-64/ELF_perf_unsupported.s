# REQUIRES: native && x86_64-linux
# UNSUPPORTED: wsl1, msan
#
# RUN: llvm-mc -triple=x86_64-linux -filetype=obj %s -o %t-unknown.o
# RUN: %python %S/../Inputs/check-perf-jitdump.py --no-unwind-for main --run \
# RUN:   llvm-jitlink -perf-support %t-unknown.o | FileCheck %s
# RUN: llvm-mc -triple=x86_64-linux -filetype=obj -defsym=BAD_EXPR=1 %s -o %t-expr.o
# RUN: %python %S/../Inputs/check-perf-jitdump.py --no-unwind-for main --run \
# RUN:   llvm-jitlink -perf-support %t-expr.o | FileCheck %s
#
# The profiler cannot normalize these CFI programs. That must not prevent
# materialization, the call to puts, or emission of the CODE_LOAD record.
# CHECK: EXECUTED
# CHECK: main: code load without unwind
# CHECK: supported: valid per-function unwind record

        .text
        .globl main
        .type main,@function
main:
        .cfi_startproc
.ifdef BAD_EXPR
        # DW_CFA_def_cfa_expression: DW_OP_addr requires relocation context.
        .cfi_escape 0x0f, 0x09, 0x03, 0, 0, 0, 0, 0, 0, 0, 0
.else
        # An unknown extended CFI opcode.
        .cfi_escape 0x17
.endif
        pushq %rax
        .cfi_adjust_cfa_offset 8
        leaq .Lmessage(%rip), %rdi
        callq puts
        popq %rcx
        .cfi_adjust_cfa_offset -8
        xorl %eax, %eax
        retq
        .cfi_endproc
        .size main, .-main

        .globl supported
        .type supported,@function
supported:
        .cfi_startproc
        xorl %eax, %eax
        retq
        .cfi_endproc
        .size supported, .-supported

        .section .rodata
.Lmessage:
        .asciz "EXECUTED"
        .section .note.GNU-stack,"",@progbits
