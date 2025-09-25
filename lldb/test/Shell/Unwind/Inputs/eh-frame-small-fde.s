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
        # Make the FDE entry start 16 bytes later than the actual function. The
        # size is chosen such that it is larger than the size of the FDE entry.
        # This allows us to test that we're using the correct offset for
        # unwinding (we'll stop 21 bytes into the function, but only 5 bytes
        # into the FDE).
        .nops 16
        .cfi_startproc
        .cfi_register %rip, %r13
        call    bar
        addl    $1, %eax
        movq    %r13, %r14
        .cfi_register %rip, %r14
        movq    $0, %r13
        jmp     *%r14 # Return
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
