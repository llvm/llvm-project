# REQUIRES: asserts
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t %s
# RUN: llvm-jitlink -debug-only=orc -noexec -abs _external_func=0x1 \
# RUN:   -entry=_foo %t 2>&1 | FileCheck %s
#
# Check that simplification eliminates dependencies on symbols in this unit,
# and correctly propagates dependencies on symbols outside the unit (including
# via locally scoped symbols). In this test _baz depends on _foo indirectly via
# the local symbol _bar. Initially we expect _baz to depend on _foo, and _foo
# on _external_func, after simplification we expect both to depend on
# _external_func only.	

# CHECK: In main emitting {{.*}}_foo{{.*}}
# CHECK-NEXT: Initial dependencies:
# CHECK-DAG: Symbols: { _foo }, Dependencies: { (main, { _external_func }) }
# CHECK-DAG: Symbols: { _baz }, Dependencies: { (main, { _foo }) }
# CHECK: Simplified dependencies:
# CHECK-DAG: Symbols: { _foo }, Dependencies: { (main, { _external_func }) }
# CHECK-DAG: Symbols: { _baz }, Dependencies: { (main, { _external_func }) }

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
