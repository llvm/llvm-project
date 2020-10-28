// The purpose of this test is to verify that we produce relocations for
// references to functions.  Failing to do so might cause pointer-to-function
// equality to fail if /INCREMENTAL links are used.

// RUN: llvm-mc -filetype=obj -incremental-linker-compatible -triple i686-pc-win32 %s | llvm-readobj -S - | FileCheck %s
// RUN: llvm-mc -filetype=obj -incremental-linker-compatible -triple x86_64-pc-win32 %s | llvm-readobj -S - | FileCheck %s

	.def	 _foo;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	_foo
	.align	16, 0x90
_foo:                                   # @foo
# %bb.0:                                # %e
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
# %bb.0:                                # %e
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
# %bb.0:                                # %e
	subl	$4, %esp
Ltmp0:
	call	_baz
	addl	$4, %esp
	ret

// CHECK:     Sections [
// CHECK: RelocationCount: 1
