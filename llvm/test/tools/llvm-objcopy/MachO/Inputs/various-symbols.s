# static int i; // A local symbol.
# int f(void) { return i; } // An external symbol.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14
	.globl	_f                      ## -- Begin function f
	.p2align	4, 0x90
_f:                                     ## @f
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	_i(%rip), %eax
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
.zerofill __DATA,__bss,_i,4,2           ## @i

.subsections_via_symbols
