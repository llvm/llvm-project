# RUN: llvm-mc -triple=arm64e-apple-macosx -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=orc -noexec \
# RUN:              -abs _foo=0x1 %t.o 2>&1 \
# RUN:              | FileCheck %s
#
# Ensure that we don't lose dependence tracking information when ptrauth edges
# are lowered: _main should still depend on _foo.
#
# REQUIRES: asserts

# CHECK:    Symbols: { _main }, Dependencies: { (main, { _foo }) }

	.section	__TEXT,__text,regular,pure_instructions

	.section	__DATA,__data
	.globl	_main
	.p2align	3, 0x0
_main:
	.quad	_foo@AUTH(ia,0)

.subsections_via_symbols
