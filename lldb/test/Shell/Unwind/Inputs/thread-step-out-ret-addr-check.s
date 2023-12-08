        .text
        .globl  asm_main
asm_main:
        sub $0x8, %rsp
        movq $0, (%rsp)
        push %rsp
        jmp nonstandard_stub

# Takes a single pointer argument via the stack, which is nonstandard for x64.
# Executing 'thread step-out' here will initially attempt to write a
# breakpoint to that stack address, but should fail because of the executable
# memory check.
        .globl nonstandard_stub
nonstandard_stub:
        mov (%rsp), %rdi
        mov (%rdi), %rsi
        add $1, %rsi
        mov %rsi, (%rdi)

        add $0x10, %rsp
        ret

#ifdef __linux__
        .section .note.GNU-stack,"",@progbits
#endif
