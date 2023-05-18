// RUN: llvm-mc %s -o %t -filetype=obj -triple=x86_64-apple-darwin
// RUN: llvm-nm -n %t | FileCheck %s
// CHECK: 0000000000000000 t _f
// CHECK: 0000000000000004 C _a

_f:
	retq

	.comm	_a,4
