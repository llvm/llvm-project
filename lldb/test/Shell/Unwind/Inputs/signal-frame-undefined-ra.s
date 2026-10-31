        .att_syntax
        .text
        .globl  asm_main
asm_main:
        .cfi_startproc
        callq   terminal
        xorl    %eax, %eax
        retq
        .cfi_endproc

        .globl  terminal
terminal:
        .cfi_startproc
        .cfi_signal_frame
        .cfi_def_cfa %rsp, 8
        .cfi_undefined %rip
        int3
        retq
        .cfi_endproc
