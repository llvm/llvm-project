	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14
	.globl	__Z1fi                  ## -- Begin function _Z1fi
	.p2align	4, 0x90
__Z1fi:                                 ## @_Z1fi
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
	movl	_x(%rip), %edi
	addl	-4(%rbp), %edi
	movl	%edi, %eax
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.globl	_main                   ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movl	$2, %edi
	movl	$0, -4(%rbp)
	callq	__Z1fi
	addq	$16, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__DATA,__data
	.globl	_x                      ## @x
	.p2align	2
_x:
	.long	1                       ## 0x1


.subsections_via_symbols
