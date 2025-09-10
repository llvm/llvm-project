	.file	"t.ll"
	.text
	.globl	f                               # -- Begin function f
	.p2align	4
	.type	f,@function
f:                                      # @f
	.cfi_startproc
# %bb.0:                                # %entry
                                        # kill: def $edi killed $edi def $rdi
                                        # kill: def $esi killed $esi def $rsi
	leal	(%rsi,%rdi), %eax
	addl	%edx, %eax
	retq
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
