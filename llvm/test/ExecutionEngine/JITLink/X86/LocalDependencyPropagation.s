# REQUIRES: asserts
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t %s
# RUN: llvm-jitlink -debug-only=orc -noexec -abs _external_func=0x1 \
# RUN:   -entry=_foo %t 2>&1 | FileCheck %s
#
# Verify that symbol dependencies are correctly propagated through local
# symbols: _baz depends on _foo indirectly via the local symbol _bar. We expect
# _baz to depend on _foo, and _foo on _external_func.

# CHECK-DAG: In main adding dependencies for _foo: { (main, { _external_func }) }
# CHECK-DAG: In main adding dependencies for _baz: { (main, { _foo }) }

        .section	__TEXT,__text,regular,pure_instructions

	.globl	_foo
	.p2align	4, 0x90
_foo:
	jmp	_external_func

	.p2align	4, 0x90
_bar:

	jmp	_foo

	.globl	_baz
	.p2align	4, 0x90
_baz:

	jmp	_bar

.subsections_via_symbols
