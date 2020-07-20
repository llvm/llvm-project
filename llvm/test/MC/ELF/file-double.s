// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj --symbols - | FileCheck %s

// Test that a STT_FILE symbol and a symbol of the same name can coexist.

.file "foo.c"
.file "bar.c"
	.globl foo.c
foo.c:

	.globl bar.c
bar.c:

// CHECK:        Symbol {
// CHECK:          Name: foo.c
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: File
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Absolute (0xFFF1)
// CHECK-NEXT:   }
// CHECK:          Name: bar.c
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: File
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Absolute (0xFFF1)
// CHECK-NEXT:   }
// CHECK:        Symbol {
// CHECK:        Name: bar.c
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
// CHECK:        Symbol {
// CHECK:        Name: foo.c
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }
