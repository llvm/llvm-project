// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj --symbols - | FileCheck %s

	.weakref	bar,foo
	call	bar@PLT

// CHECK:        Symbol {
// CHECK:          Name: foo
// CHECK-NEXT:     Value:
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Binding: Weak
// CHECK-NEXT:     Type:
// CHECK-NEXT:     Other:
// CHECK-NEXT:     Section:
// CHECK-NEXT:   }
