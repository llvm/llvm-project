// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o - | llvm-readobj --symbols -r | FileCheck %s

	.data
	.globl	test1_foo
test1_foo:
        .long 42

        .globl test1_zed
test1_zed = test1_foo + 1

// CHECK:      Symbol {
// CHECK:        Name: test1_zed
// CHECK-NEXT:   Value: 1
// CHECK-NEXT:   Section: .data
// CHECK-NEXT:   BaseType: Null
// CHECK-NEXT:   ComplexType: Null
// CHECK-NEXT:   StorageClass: External
// CHECK-NEXT:   AuxSymbolCount: 0
// CHECK-NEXT: }
