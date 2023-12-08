    .text
    .globl func
    .def func; .scl 2; .type 32; .endef
    .seh_proc func
func:
    .seh_pushframe @code
    subq $24, %rsp
    .seh_stackalloc 24
    movq %rsi, 16(%rsp)
    .seh_savereg %rsi, 16
    movups %xmm8, (%rsp)
    .seh_savexmm %xmm8, 0
    pushq %rbx
    .seh_pushreg %rbx
    mov %rsp, %rbx
    .seh_setframe 3, 0
    .seh_endprologue
    .seh_handler __C_specific_handler, @except
    .seh_handlerdata
    .long 0
    .text
    .seh_startchained
    .seh_endprologue
    .seh_endchained
    lea (%rbx), %rsp
    pop %rbx
    addq $24, %rsp
    ret
    .seh_endproc

// Test emission of small functions.
    .globl smallFunc
    .def smallFunc; .scl 2; .type 32; .endef
    .seh_proc smallFunc
smallFunc:
    ret
    .seh_endproc

// Function with big stack allocation.
    .globl allocFunc
    .def allocFunc; .scl 2; .type 32; .endef
    .seh_proc allocFunc
allocFunc:
    .seh_pushframe @code
    subq $65520, %rsp
    .seh_stackalloc 65520
    sub $8454128, %rsp
    .seh_stackalloc 8454128
    .seh_endprologue
    add $8454128, %rsp
    addq $65520, %rsp
    ret
    .seh_endproc
