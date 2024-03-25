        .text
        .globl  asm_main
asm_main:
        .cfi_startproc
        cmpb $0x0, g_hard_abort(%rip)
        jne .L

        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset 6, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register 6
        callq   abort_function
.L:
        .cfi_def_cfa 7, 8
        .cfi_restore 6
        int3
        ud2
        .cfi_endproc

        .globl abort_function
abort_function:
        .cfi_startproc
        ud2
        .cfi_endproc

	.data
	.globl  g_hard_abort
g_hard_abort:
	.byte   1
