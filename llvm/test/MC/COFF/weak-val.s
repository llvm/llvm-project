// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -o %t.o
// RUN: llvm-readobj --symbols %t.o | FileCheck %s

// test that b and .weak.b have the correct values.

        .data
.long 42
        .weak b
b:
        .long   42

// CHECK:      Symbol {
// CHECK:        Name: b
// CHECK-NEXT:   Value: 0
// CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:   BaseType: Null (0x0)
// CHECK-NEXT:   ComplexType: Null (0x0)
// CHECK-NEXT:   StorageClass: WeakExternal (0x69)
// CHECK-NEXT:   AuxSymbolCount: 1
// CHECK-NEXT:   AuxWeakExternal {
// CHECK-NEXT:     Linked: .weak.b.default (8)
// CHECK-NEXT:     Search: Alias (0x3)
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: .weak.b.default{{$}}
// CHECK-NEXT:   Value: 4
// CHECK-NEXT:   Section: .data (2)
// CHECK-NEXT:   BaseType: Null (0x0)
// CHECK-NEXT:   ComplexType: Null (0x0)
// CHECK-NEXT:   StorageClass: External (0x2)
// CHECK-NEXT:   AuxSymbolCount: 0
// CHECK-NEXT: }
