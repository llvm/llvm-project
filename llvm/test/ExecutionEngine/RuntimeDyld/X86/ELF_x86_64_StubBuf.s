# RUN: llvm-mc -triple=x86_64-apple-macosx10.10.0 -filetype=obj -o %T/test_ELF_x86_64_StubBuf.o %s
# RUN: llvm-rtdyld -print-alloc-requests -triple=x86_64-pc-linux -dummy-extern _g=196608 -verify %T/test_ELF_x86_64_StubBuf.o

# Compiled from Inputs/ELF/ELF_x86_64_StubBuf.ll

# CHECK: allocateCodeSection(Size = 42, Alignment = 16, SectionName = __text)

	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 10
	.globl	_f
	.align	4, 0x90
_f:                                     ## @f
	.cfi_startproc
## BB#0:                                ## %entry
	pushq	%rax
Ltmp0:
	.cfi_def_cfa_offset 16
	callq	_g
	callq	_g
	callq	_g
	popq	%rax
	retq
	.cfi_endproc


.subsections_via_symbols
