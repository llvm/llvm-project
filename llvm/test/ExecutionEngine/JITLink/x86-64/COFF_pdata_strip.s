# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN:
# RUN: llvm-jitlink -abs __ImageBase=0xdeadbeaf -noexec %t \
# RUN: -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN: -show-graphs='.*' -noexec 2>&1 | FileCheck %s
#
# Check that basic seh frame of dead block is dead-stripped out
#
# CHECK: section .func:
# CHECK-EMPTY:
# CHECK-NEXT: section .pdata:
# CHECK-EMPTY:
# CHECK: section .xdata:
# CHECK-EMPTY:

	.text

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
	retq

	.section .func

    .def	func;
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
func:
	.seh_proc func
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movl	$0, 36(%rsp)
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
