// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj --symbols - | FileCheck %s

foo:
bar = foo

        .globl	foo2
foo2 = bar2

foo3:
	.globl	bar3
bar3 = foo3


// Test that bar4  and bar 5 are also functions and have the same value as foo4.
	.byte 0
	.type	foo4,@function
foo4:
bar4 = foo4
bar5 = bar4

        .long foo2

// Test that bar6 is a function that doesn't have the same value as foo4.
bar6 = bar5
bar6:

// Test that indirect local aliases do not appear as symbols.
.data
.Llocal:

.text
leaq .Llocal1(%rip), %rdi
.Llocal1 = .Llocal2
.Llocal2 = .Llocal

// CHECK:      Symbols [
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name:  (0)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local (0x0)
// CHECK-NEXT:     Type: None (0x0)
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: bar
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: bar4
// CHECK-NEXT:     Value: 0x1
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Function
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: bar5
// CHECK-NEXT:     Value: 0x1
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Function
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: bar6
// CHECK-NEXT:     Value: 0x5
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: foo
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: foo3
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: foo4
// CHECK-NEXT:     Value: 0x1
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Function
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: .data (0)
// CHECK-NOT: Symbol {
// CHECK:        }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: bar2
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: bar3
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK-NEXT: ]
