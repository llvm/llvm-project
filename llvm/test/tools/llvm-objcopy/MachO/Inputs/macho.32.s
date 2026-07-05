	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14
	.globl	__Z1fi                  ## -- Begin function _Z1fi
	.p2align	4, 0x90
__Z1fi:                                 ## @_Z1fi
	.cfi_startproc
## %bb.0:
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset %ebp, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register %ebp
	pushl	%eax
	calll	L0$pb
L0$pb:
	popl	%eax
	movl	8(%ebp), %ecx
	movl	_x-L0$pb(%eax), %eax
	addl	8(%ebp), %eax
	movl	%ecx, -4(%ebp)          ## 4-byte Spill
	addl	$4, %esp
	popl	%ebp
	retl
	.cfi_endproc
                                        ## -- End function
	.globl	_main                   ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset %ebp, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register %ebp
	subl	$24, %esp
	movl	$2, %eax
	movl	$0, -4(%ebp)
	movl	$2, (%esp)
	movl	%eax, -8(%ebp)          ## 4-byte Spill
	calll	__Z1fi
	addl	$24, %esp
	popl	%ebp
	retl
	.cfi_endproc
                                        ## -- End function
	.section	__DATA,__data
	.globl	_x                      ## @x
	.p2align	2
_x:
	.long	1                       ## 0x1


.subsections_via_symbols
