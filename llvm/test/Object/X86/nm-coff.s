// RUN: llvm-mc %s -o %t -filetype=obj -triple=x86_64-pc-win32
// RUN: llvm-nm --undefined-only %t | FileCheck %s
// CHECK: w foo

g:
	movl	foo(%rip), %eax
	retq

	.weak	foo
