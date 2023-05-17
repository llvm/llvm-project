        .globl    call_func
        .def      call_func;   .scl    2;      .type   32;     .endef
        .seh_proc call_func
call_func:
        subq    $32, %rsp
        .seh_stackalloc 32
        .seh_endprologue
        call    realign_stack
        addq    $32, %rsp
        ret
        .seh_endproc

        .globl    realign_stack
        .def      realign_stack;   .scl    2;      .type   32;     .endef
        .seh_proc realign_stack
realign_stack:
        subq    $32, %rsp
        .seh_stackalloc 32
        .seh_endprologue
        movq    %rcx, %rax
        movl    %edx, %ecx
        call    *%rax
        addq    $32, %rsp
        ret
        .seh_endproc
