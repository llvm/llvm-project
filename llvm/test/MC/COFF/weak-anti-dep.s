// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | llvm-readobj --symbols - | FileCheck %s

// CHECK:       Symbol {
// CHECK-NEXT:    Name: .text
// CHECK:       Symbol {
// CHECK-NEXT:    Name: .data
// CHECK:       Symbol {
// CHECK-NEXT:    Name: .bss

.weak_anti_dep a
a = b

// CHECK:       Symbol {
// CHECK-NEXT:    Name: a
// CHECK-NEXT:    Value: 0
// CHECK-NEXT:    Section: IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:    BaseType: Null (0x0)
// CHECK-NEXT:    ComplexType: Null (0x0)
// CHECK-NEXT:    StorageClass: WeakExternal (0x69)
// CHECK-NEXT:    AuxSymbolCount: 1
// CHECK-NEXT:    AuxWeakExternal {
// CHECK-NEXT:      Linked: b (8)
// CHECK-NEXT:      Search: AntiDependency (0x4)
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: b
// CHECK-NEXT:    Value: 0
// CHECK-NEXT:    Section: IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:    BaseType: Null (0x0)
// CHECK-NEXT:    ComplexType: Null (0x0)
// CHECK-NEXT:    StorageClass: External (0x2)
// CHECK-NEXT:    AuxSymbolCount: 0
// CHECK-NEXT:  }


.weak_anti_dep r1
.weak_anti_dep r2
r1 = r2
r2 = r1


// CHECK:       Symbol {
// CHECK-NEXT:    Name: r1
// CHECK-NEXT:    Value: 0
// CHECK-NEXT:    Section: IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:    BaseType: Null (0x0)
// CHECK-NEXT:    ComplexType: Null (0x0)
// CHECK-NEXT:    StorageClass: WeakExternal (0x69)
// CHECK-NEXT:    AuxSymbolCount: 1
// CHECK-NEXT:    AuxWeakExternal {
// CHECK-NEXT:      Linked: r2 (11)
// CHECK-NEXT:      Search: AntiDependency (0x4)
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: r2
// CHECK-NEXT:    Value: 0
// CHECK-NEXT:    Section: IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:    BaseType: Null (0x0)
// CHECK-NEXT:    ComplexType: Null (0x0)
// CHECK-NEXT:    StorageClass: WeakExternal (0x69)
// CHECK-NEXT:    AuxSymbolCount: 1
// CHECK-NEXT:    AuxWeakExternal {
// CHECK-NEXT:      Linked: r1 (9)
// CHECK-NEXT:      Search: AntiDependency (0x4)
// CHECK-NEXT:    }
// CHECK-NEXT:  }
