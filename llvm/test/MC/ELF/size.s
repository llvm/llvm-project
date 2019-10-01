// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux < %s | llvm-readobj --symbols | FileCheck %s

foo:
bar = .
	.size	foo, . - bar + 42

// CHECK:       Symbol {
// CHECK:         Name: foo
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 42
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
