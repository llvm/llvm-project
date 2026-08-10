// RUN: llvm-mc -triple x86_64-unknown-windows-msvc -filetype obj -o - %s | llvm-readobj --symbols - | FileCheck %s

g:
	movl	foo(%rip), %eax
	retq

	.weak	foo

// CHECK: Symbol {
// CHECK:   Name: foo
// CHECK:   Section: IMAGE_SYM_UNDEFINED (0)
// CHECK:   StorageClass: WeakExternal (0x69)
// CHECK: }

