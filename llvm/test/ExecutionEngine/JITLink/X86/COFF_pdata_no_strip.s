# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: 
# RUN: llvm-jitlink -abs __ImageBase=0xdeadbeaf -noexec %t \
# RUN: -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN: -show-graph -noexec 2>&1 | FileCheck %s
#
# Check that basic seh frame inside pdata of alive function is not dead-stripped out.
# CHECK: section .xdata:
# CHECK-EMPTY:
# CHECK-NEXT: block 0xfff00000 size = 0x00000008, align = 4, alignment-offset = 0
# CHECK-NEXT: symbols:
# CHECK-NEXT:   0xfff00000 (block + 0x00000000): size: 0x00000008, linkage: strong, scope: local, live  -   .xdata

	.text

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
.seh_proc main
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movl	$0, 36(%rsp)
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
