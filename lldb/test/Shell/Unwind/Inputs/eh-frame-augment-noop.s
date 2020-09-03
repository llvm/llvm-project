# A function whose prologue and epilogue are described perfectly.  eh_frame
# augmentation machinery should detect that no augmentation is needed and use
# eh_frame directly.

        .text
        .globl  foo
foo:
        .cfi_startproc
        pushq   %rax
        .cfi_def_cfa_offset 16
        int3
        pop %rcx
        .cfi_def_cfa_offset 8
        retq
        .cfi_endproc

        .globl asm_main
asm_main:
        .cfi_startproc
        callq foo
        retq
        .cfi_endproc
