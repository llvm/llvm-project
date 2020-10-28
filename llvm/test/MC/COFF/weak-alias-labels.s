// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | llvm-readobj --symbols - | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | llvm-readobj --symbols - | FileCheck %s

// When making weak references to labels/procedures, we reference them directly
// if they have global symbols; otherwise, we need to create a global symbol for
// the reference to resolve to.

    .text

    .globl proc1
proc1:
    ret

proc2:
    ret
// CHECK:      Symbol {
// CHECK:        Name: proc2
// CHECK-NEXT:   Value: [[PROC2_VALUE:[0-9]+]]
// CHECK-NEXT:   Section: [[PROC2_SECTION:.*]]
// CHECK-NEXT:   BaseType: Null
// CHECK-NEXT:   ComplexType: Null
// CHECK-NEXT:   StorageClass: Static
// CHECK-NEXT:   AuxSymbolCount: 0
// CHECK-NEXT: }

    .weak t1
t1 = proc1

// CHECK:      Symbol {
// CHECK:        Name: t1
// CHECK-NEXT:   Value: 0
// CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED
// CHECK-NEXT:   BaseType: Null
// CHECK-NEXT:   ComplexType: Null
// CHECK-NEXT:   StorageClass: WeakExternal
// CHECK-NEXT:   AuxSymbolCount: 1
// CHECK-NEXT:   AuxWeakExternal {
// CHECK-NEXT:     Linked: proc1
// CHECK-NEXT:     Search: Alias
// CHECK-NEXT:   }
// CHECK-NEXT: }

    .weak t2
t2 = proc2

// CHECK:      Symbol {
// CHECK:        Name: t2
// CHECK-NEXT:   Value: 0
// CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED
// CHECK-NEXT:   BaseType: Null
// CHECK-NEXT:   ComplexType: Null
// CHECK-NEXT:   StorageClass: WeakExternal
// CHECK-NEXT:   AuxSymbolCount: 1
// CHECK-NEXT:   AuxWeakExternal {
// CHECK-NEXT:     Linked: .weak.t2.default
// CHECK-NEXT:     Search: Alias
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK:      Symbol {
// CHECK:        Name: .weak.t2.default
// CHECK-NEXT:   Value: [[PROC2_VALUE]]
// CHECK-NEXT:   Section: [[PROC2_SECTION]]
// CHECK-NEXT:   BaseType: Null
// CHECK-NEXT:   ComplexType: Null
// CHECK-NEXT:   StorageClass: External
// CHECK-NEXT:   AuxSymbolCount: 0
// CHECK-NEXT: }

    .weak t3
t3 = foo

// CHECK:      Symbol {
// CHECK:        Name: t3
// CHECK-NEXT:   Value: 0
// CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED
// CHECK-NEXT:   BaseType: Null
// CHECK-NEXT:   ComplexType: Null
// CHECK-NEXT:   StorageClass: WeakExternal
// CHECK-NEXT:   AuxSymbolCount: 1
// CHECK-NEXT:   AuxWeakExternal {
// CHECK-NEXT:     Linked: foo
// CHECK-NEXT:     Search: Alias
// CHECK-NEXT:   }
// CHECK-NEXT: }

    .weak t4
t4 = bar

    .globl bar
bar:
    ret

// CHECK:      Symbol {
// CHECK:        Name: t4
// CHECK-NEXT:   Value: 0
// CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED
// CHECK-NEXT:   BaseType: Null
// CHECK-NEXT:   ComplexType: Null
// CHECK-NEXT:   StorageClass: WeakExternal
// CHECK-NEXT:   AuxSymbolCount: 1
// CHECK-NEXT:   AuxWeakExternal {
// CHECK-NEXT:     Linked: bar
// CHECK-NEXT:     Search: Alias
// CHECK-NEXT:   }
// CHECK-NEXT: }

    .weak t5
t5 = t2

// CHECK:      Symbol {
// CHECK:        Name: t5
// CHECK-NEXT:   Value: 0
// CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED
// CHECK-NEXT:   BaseType: Null
// CHECK-NEXT:   ComplexType: Null
// CHECK-NEXT:   StorageClass: WeakExternal
// CHECK-NEXT:   AuxSymbolCount: 1
// CHECK-NEXT:   AuxWeakExternal {
// CHECK-NEXT:     Linked: t2
// CHECK-NEXT:     Search: Alias
// CHECK-NEXT:   }
// CHECK-NEXT: }
