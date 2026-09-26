// Check that a trivial MachO arm64 object can be linked and run under ogre.
//
// RUN: %{mc} -o %t.o %s
// RUN: %{jit} -show-jit-result %t.o | FileCheck %s

// CHECK: JIT result: 0

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	2
_main:
	mov	w0, #0
	ret

	.subsections_via_symbols
