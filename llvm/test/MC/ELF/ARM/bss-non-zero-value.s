// RUN: not llvm-mc -filetype=obj -triple arm-linux-gnu %s -o %t 2>%t.out
// RUN: FileCheck --input-file=%t.out %s
// CHECK: non-zero initializer found in section '.bss'
	.bss
	.globl	a
	.align	2
a:
	.long	1
	.size	a, 4
