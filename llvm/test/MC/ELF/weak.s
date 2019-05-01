// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj --symbols | FileCheck %s

// Test that this produces a weak undefined symbol.

	.weak	foo
        .long   foo

// And that bar is after all local symbols and has non-zero value.
        .weak bar
bar:

// CHECK:        Symbol {
// CHECK:          Name: bar
// CHECK-NEXT:     Value: 0x4
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Weak
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK:        Symbol {
// CHECK:          Name: foo
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Weak
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
