# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin10.0 %s | llvm-objdump --unwind-info - | FileCheck %s

# Check that we emit the right encoding for the no-personality case.

# CHECK: Contents of __compact_unwind section:
# CHECK:   Entry at offset {{.+}}
# CHECK:     start:                {{.+}} __Z3barPi
# CHECK:     length:               {{.+}}
# CHECK:     compact encoding:     0x01000000
# CHECK:   Entry at offset {{.+}}
# CHECK:     start:                {{.+}} __Z3foov
# CHECK:     length:               {{.+}}
# CHECK:     compact encoding:     0x01000000
# CHECK:   Entry at offset {{.+}}
# CHECK:     start:                {{.+}} _main
# CHECK:     length:               0x1c

        .section	__TEXT,__text,regular,pure_instructions
	.globl	__Z3barPi        
	.p2align	4, 0x90
__Z3barPi:   
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	popq	%rbp
	retq
	.cfi_endproc

	.globl	__Z3foov        
	.p2align	4, 0x90
__Z3foov:                        
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	leaq	-4(%rbp), %rdi
	callq	__Z3barPi
	addq	$16, %rsp
	popq	%rbp
	retq
	.cfi_endproc

	.globl	_main          
	.p2align	4, 0x90
_main:                       
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movl	$0, -4(%rbp)
	callq	__Z3foov
	xorl	%eax, %eax
	addq	$16, %rsp
	popq	%rbp
	retq
	.cfi_endproc

.subsections_via_symbols
