# RUN: yaml2obj -o %T/libfoo.dylib %S/Inputs/libFooUniversalDylib.yaml
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj \
# RUN:     -o %T/MachO_foo-in-weak-dylib.o %s
# RUN: llvm-jitlink -noexec %T/MachO_foo-in-weak-dylib.o \
# RUN:     -weak_library %T/libfoo.dylib
#
# Check that -weak_library supports universal binaries.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0	sdk_version 15, 4
	.globl	_main
	.p2align	4, 0x90
_main:
	pushq	%rbp
	movq	%rsp, %rbp
	cmpl	$101, %edi
	jl	LBB0_1
	popq	%rbp
	jmp	_foo
LBB0_1:
	xorl	%eax, %eax
	popq	%rbp
	retq

.subsections_via_symbols
