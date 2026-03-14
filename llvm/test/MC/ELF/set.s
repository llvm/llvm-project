// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj --symbols - | FileCheck  %s

// Test that we emit the correct value.

.set kernbase,0xffffffff80000000

// CHECK:        Symbol {
// CHECK:          Name: kernbase
// CHECK-NEXT:     Value: 0xFFFFFFFF80000000
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Absolute (0xFFF1)
// CHECK-NEXT:   }

// Test that we accept .set of a symbol after it has been used in a statement.

        jmp foo
        .set foo, bar

// or a .quad

        .quad	foo2
	.set	foo2,bar2

// Test that there is an undefined reference to bar
// CHECK:        Symbol {
// CHECK:          Name: bar
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined (0x0)
// CHECK-NEXT:   }
