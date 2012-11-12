// The purpose of this test is to verify that we do not produce unneeded
// relocations when symbols are in the same section and we know their offset.

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | coff-dump.py | FileCheck %s
// I WOULD RUN, BUT THIS FAILS: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | coff-dump.py | FileCheck %s

	.def	 _foo;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	_foo
	.align	16, 0x90
_foo:                                   # @foo
# BB#0:                                 # %e
	.align	16, 0x90
LBB0_1:                                 # %i
                                        # =>This Inner Loop Header: Depth=1
	jmp	LBB0_1

	.def	 _bar;
	.scl	2;
	.type	32;
	.endef
	.globl	_bar
	.align	16, 0x90
_bar:                                   # @bar
# BB#0:                                 # %e
	.align	16, 0x90
LBB1_1:                                 # %i
                                        # =>This Inner Loop Header: Depth=1
	jmp	LBB1_1

	.def	 _baz;
	.scl	2;
	.type	32;
	.endef
	.globl	_baz
	.align	16, 0x90
_baz:                                   # @baz
# BB#0:                                 # %e
	subl	$4, %esp
Ltmp0:
	calll	_baz
	addl	$4, %esp
	ret

// CHECK:     Sections = [
// CHECK-NOT: NumberOfRelocations = {{[^0]}}
// CHECK:     Symbols = [
