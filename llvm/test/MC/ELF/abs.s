// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj --symbols | FileCheck %s

// Test that zed will be an ABS symbol

.Lfoo:
.Lbar:
        zed = .Lfoo - .Lbar

// CHECK:        Symbol {
// CHECK:          Name: zed
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Absolute (0xFFF1)
// CHECK-NEXT:   }
