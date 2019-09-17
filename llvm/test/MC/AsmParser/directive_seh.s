# RUN: llvm-mc -triple x86_64-pc-win32 %s | FileCheck %s

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
    lea (%rbx), %rsp
    pop %rbx
    addq $24, %rsp
    ret
    .seh_endproc
# CHECK: .seh_endproc
