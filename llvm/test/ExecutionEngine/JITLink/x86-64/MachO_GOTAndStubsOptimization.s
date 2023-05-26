# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-apple-macos10.9 -filetype=obj \
# RUN:     -o %t/helper.o %S/Inputs/MachO_GOTAndStubsOptimizationHelper.s
# RUN: llvm-mc -triple=x86_64-apple-macos10.9 -filetype=obj \
# RUN:     -o %t/testcase.o %s
# RUN: llvm-jitlink -noexec -slab-allocate 64Kb -slab-page-size 4096 \
# RUN:     -entry=bypass_stub -check %s %t/testcase.o %t/helper.o
#
# Test that references to in-range GOT and stub targets can be bypassed.
# The helper file contains a function that uses the GOT for _x, and this file
# contains an external call to that function. By slab allocating the JIT memory
# we can ensure that the references and targets will be in-range of one another,
# which should cause both the GOT load and stub to be bypassed.

        .section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 14
	.globl bypass_stub
	.p2align	4, 0x90

# jitlink-check: decode_operand(bypass_got, 4) = _x - next_pc(bypass_got)
# jitlink-check: decode_operand(bypass_stub, 0) = bypass_got - next_pc(bypass_stub)
bypass_stub:
	callq	bypass_got

	.section	__DATA,__data
	.globl	_x
	.p2align	2
_x:
	.long	42

.subsections_via_symbols
