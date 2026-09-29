# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-freebsd -filetype=obj -o %t/testsmall_x86-64.o %s
# RUN: llvm-rtdyld -triple=x86_64-unknown-freebsd -verify -check=%s %t/testsmall_x86-64.o

	.globl	foo
	.align	4, 0x90
foo:
        retq

	.globl	main
	.align	4, 0x90
main:
# Test PC-rel branch.
# rtdyld-check: decode_operand(insn1, 0) = foo - next_pc(insn1)
insn1:
        callq	foo
