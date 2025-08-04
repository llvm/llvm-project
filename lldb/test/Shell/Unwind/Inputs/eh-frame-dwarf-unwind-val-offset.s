        .text
        .globl  bar
bar:
        .cfi_startproc
        leal    (%edi, %edi), %eax
        ret
        .cfi_endproc

        .globl  foo
        # This function uses a non-standard calling convention (return address
        # needs to be adjusted) to force lldb to use eh_frame/debug_frame
        # instead of reading the code directly.
foo:
        .cfi_startproc
        .cfi_escape 0x16, 0x10, 0x06, 0x38, 0x1c, 0x06, 0x08, 0x47, 0x1c
        # Clobber r12 and record that it's reconstructable from CFA
        .cfi_val_offset %r12, 32
        movq    $0x456, %r12
        call    bar
        addl    $1, %eax
        # Reconstruct %r12
        movq    %rsp, %r12
        addq    %r12, 8
        popq    %rdi
        subq    $0x47, %rdi
        jmp     *%rdi # Return
        .cfi_endproc

        .globl  asm_main
asm_main:
        .cfi_startproc
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
        movq    %rsp, %r12
        addq    $32, %r12
        .cfi_def_cfa_register %rbp
        movl    $47, %edi

        # Non-standard calling convention. The real return address must be
        # decremented by 0x47.
        leaq    0x47+1f(%rip), %rax
        pushq   %rax
        jmp     foo # call
1:
        popq    %rbp
        .cfi_def_cfa %rsp, 8
        ret
        .cfi_endproc
