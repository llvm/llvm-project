        .text

        .type   bar, @function
bar:
        .cfi_startproc
        leal    (%edi, %edi), %eax
        ret
        .cfi_endproc
        .size   bar, .-bar

        .type   foo, @function
foo:
        nop # Make the FDE entry start one byte later than the actual function.
        .cfi_startproc
        .cfi_register %rip, %r13
        call    bar
        addl    $1, %eax
        jmp     *%r13 # Return
        .cfi_endproc
        .size   foo, .-foo

        .globl  main
        .type   main, @function
main:
        .cfi_startproc
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset 6, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register 6
        movl    $47, %edi

        # Non-standard calling convention. Put return address in r13.
        pushq   %r13
        leaq    1f(%rip), %r13
        jmp     foo # call
1:
        popq    %r13
        popq    %rbp
        .cfi_def_cfa 7, 8
        ret
        .cfi_endproc
        .size   main, .-main
