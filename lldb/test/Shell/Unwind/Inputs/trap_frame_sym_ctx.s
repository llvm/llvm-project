        .att_syntax
        .text
        .globl  bar
bar:
        .cfi_startproc
        leal    (%edi, %edi), %eax
        ret
        .cfi_endproc

        .globl  asm_main
asm_main:
        .cfi_startproc
        .ifdef TRAMP_CALL
        .cfi_signal_frame
        .endif
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp
        movl    $47, %edi

        .ifdef TRAMP_CALL
        # A signal-marked function can also make an ordinary call. While bar
        # is active, use the call-site row, not the terminal row at its return PC.
        call    bar
        .cfi_undefined %rip
        nop
        .cfi_offset %rip, -8
        .else
        # install tramp as return address
        # (similar to signal return trampolines on some platforms)
        leaq    tramp(%rip), %rax
        pushq   %rax
        jmp     bar # call, with return address pointing to tramp
        .endif

        popq    %rbp
        .cfi_def_cfa %rsp, 8
        ret
        .cfi_endproc

        .globl  tramp
tramp:
        .cfi_startproc
        .ifndef TRAMP_NAME_ONLY
        .cfi_signal_frame
        .endif
        # Emit cfi to line up with the frame created by asm_main
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        .cfi_def_cfa_register %rbp
        # copy asm_main's epilog to clean up the frame
        popq    %rbp
        .cfi_def_cfa %rsp, 8
        ret
        .cfi_endproc
