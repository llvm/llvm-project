// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -o %t.o
// RUN: llvm-readobj --symbols %t.o | FileCheck %s

// test that we create an external symbol for a to point to.

        .data
        .long 123
b:
        .long   42
        .weak   a
a=b

// CHECK:      Symbol {
// CHECK:        Name: b
// CHECK-NEXT:   Value: 4
// CHECK-NEXT:   Section: .data (2)
// CHECK-NEXT:   BaseType: Null (0x0)
// CHECK-NEXT:   ComplexType: Null (0x0)
// CHECK-NEXT:   StorageClass: Static (0x3)
// CHECK-NEXT:   AuxSymbolCount: 0
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: a
// CHECK-NEXT:   Value: 0
// CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:   BaseType: Null (0x0)
// CHECK-NEXT:   ComplexType: Null (0x0)
// CHECK-NEXT:   StorageClass: WeakExternal (0x69)
// CHECK-NEXT:   AuxSymbolCount: 1
// CHECK-NEXT:   AuxWeakExternal {
// CHECK-NEXT:     Linked: .weak.a.default (9)
// CHECK-NEXT:     Search: Library (0x2)
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: .weak.a.default
// CHECK-NEXT:   Value: 4
// CHECK-NEXT:   Section: .data (2)
// CHECK-NEXT:   BaseType: Null (0x0)
// CHECK-NEXT:   ComplexType: Null (0x0)
// CHECK-NEXT:   StorageClass: External (0x2)
// CHECK-NEXT:   AuxSymbolCount: 0
// CHECK-NEXT: }
