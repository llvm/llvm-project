// RUN: llvm-mc -triple i386-pc-linux-gnu %s -filetype=obj -o - | llvm-readobj --symbols | FileCheck %s

.lcomm A, 5
.lcomm B, 32 << 20

// CHECK:        Symbol {
// CHECK:          Name: A
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 5
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .bss
// CHECK-NEXT:   }
// CHECK:        Symbol {
// CHECK:          Name: B
// CHECK-NEXT:     Value: 0x5
// CHECK-NEXT:     Size: 33554432
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .bss
// CHECK-NEXT:   }
