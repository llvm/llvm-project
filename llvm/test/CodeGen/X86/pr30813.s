	.text
	.file	"/home/davide/work/llvm/test/CodeGen/X86/visitand-shift.ll"
	.globl	patatino
	.p2align	4, 0x90
	.type	patatino,@function
patatino:                               # @patatino
	.cfi_startproc
# BB#0:
                                        # implicit-def: %RAX
	movzwl	(%rax), %ecx
	movl	%ecx, %eax
                                        # implicit-def: %RDX
	movq	%rax, (%rdx)
	retq
.Lfunc_end0:
	.size	patatino, .Lfunc_end0-patatino
	.cfi_endproc


	.section	".note.GNU-stack","",@progbits
