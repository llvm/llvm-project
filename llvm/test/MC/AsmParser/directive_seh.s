# RUN: llvm-mc -triple x86_64-pc-win32 %s | FileCheck %s

#   Round trip via intel syntax printing and back.
# RUN: llvm-mc -triple x86_64-pc-win32 %s -output-asm-variant=1 | \
# RUN:     llvm-mc -triple x86_64-pc-win32 -x86-asm-syntax=intel | FileCheck %s

    .text
    .globl func
    .def func; .scl 2; .type 32; .endef
    .seh_proc func
# CHECK: .seh_proc func
func:
    .seh_pushframe @code
# CHECK: .seh_pushframe @code
    subq $24, %rsp
    .seh_stackalloc 24
# CHECK: .seh_stackalloc 24
    movq %rsi, 16(%rsp)
    .seh_savereg %rsi, 16
# CHECK: .seh_savereg %rsi, 16
    .seh_savereg 6, 16
# CHECK: .seh_savereg %rsi, 16
    movups %xmm8, (%rsp)
    .seh_savexmm %xmm8, 0
# CHECK: .seh_savexmm %xmm8, 0
    .seh_savexmm 8, 0
# CHECK: .seh_savexmm %xmm8, 0
    pushq %rbx
    .seh_pushreg %rbx
# CHECK: .seh_pushreg %rbx
    .seh_pushreg 3
# CHECK: .seh_pushreg %rbx
    mov %rsp, %rbx
    .seh_setframe 3, 0
# CHECK: .seh_setframe %rbx, 0
    .seh_endprologue
# CHECK: .seh_endprologue
    .seh_handler __C_specific_handler, @except
# CHECK: .seh_handler __C_specific_handler, @except
    .seh_handlerdata
# CHECK-NOT: .section{{.*}}.xdata
# CHECK: .seh_handlerdata
    .long 0
    .text
    .seh_startchained
    .seh_endprologue
    .seh_endchained
# CHECK: .text
# CHECK: .seh_startchained
# CHECK: .seh_endprologue
# CHECK: .seh_endchained
    .seh_startepilogue
# CHECK: .seh_startepilogue
    lea (%rbx), %rsp
    pop %rbx
    addq $24, %rsp
    .seh_endepilogue
# CHECK: .seh_endepilogue
    ret
    .seh_endproc
# CHECK: .seh_endproc

# Re-run more or less the same test, but with intel syntax. Previously LLVM
# required percent prefixing in the .seh_* directives that take registers.

    .intel_syntax noprefix
    .text
    .globl func_intel
    .def func_intel; .scl 2; .type 32; .endef
    .seh_proc func_intel
# CHECK: .seh_proc func_intel
func_intel:
    sub RSP, 24
    .seh_stackalloc 24
# CHECK: .seh_stackalloc 24
    mov [16+RSP], RSI
    .seh_savereg rsi, 16
# CHECK: .seh_savereg %rsi, 16
    .seh_savereg 6, 16
# CHECK: .seh_savereg %rsi, 16
    movups [RSP], XMM8
    .seh_savexmm XMM8, 0
# CHECK: .seh_savexmm %xmm8, 0
    .seh_savexmm 8, 0
# CHECK: .seh_savexmm %xmm8, 0
    push rbx
    .seh_pushreg rbx
# CHECK: .seh_pushreg %rbx
    .seh_pushreg 3
# CHECK: .seh_pushreg %rbx
    mov rbx, rsp
    .seh_setframe rbx, 0
# CHECK: .seh_setframe %rbx, 0
    .seh_endprologue
# CHECK: .seh_endprologue
    .seh_handler __C_specific_handler, @except
# CHECK: .seh_handler __C_specific_handler, @except
    .seh_handlerdata
# CHECK-NOT: .section{{.*}}.xdata
# CHECK: .seh_handlerdata
    .long 0
    .text
    .seh_endproc
