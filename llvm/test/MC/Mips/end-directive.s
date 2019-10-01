# RUN: llvm-mc -arch=mips -mcpu=mips32 -filetype=obj %s -o - | \
# RUN:   llvm-readobj --symbols | FileCheck %s

# Check that the assembler doesn't choke on .align between a symbol and the
# .end directive.

	.text
	.globl	a
	.p2align	2
	.type	a,@function
	.ent	a
a:
	addu	$2, $5, $4
	.align 4
	jr	$ra
	.end	a
$func_end0:
	.size	a, ($func_end0)-a

# CHECK: Name: a
# CHECK-NEXT: Value: 0x0
# CHECK-NEXT: Size: 24
