// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -o %t.o
// RUN: llvm-readobj --symbols %t.o | FileCheck %s

// Test that the weak symbol is properly undefined, while originally being
// the leader symbol for a comdat. (This can easily happen if building with
// -ffunction-sections).

        .section .text$func,"xr",one_only,func
        .weak   func
func:
        ret

// CHECK:       Symbol {
// CHECK:         Name: func
// CHECK-NEXT:    Value: 0
// CHECK-NEXT:    Section: IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:    BaseType: Null (0x0)
// CHECK-NEXT:    ComplexType: Null (0x0)
// CHECK-NEXT:    StorageClass: WeakExternal (0x69)
// CHECK-NEXT:    AuxSymbolCount: 1
// CHECK-NEXT:    AuxWeakExternal {
// CHECK-NEXT:      Linked: .weak.func.default (10)
// CHECK-NEXT:      Search: Alias (0x3)
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: .weak.func.default
// CHECK-NEXT:    Value: 0
// CHECK-NEXT:    Section: .text$func (4)
// CHECK-NEXT:    BaseType: Null (0x0)
// CHECK-NEXT:    ComplexType: Null (0x0)
// CHECK-NEXT:    StorageClass: External (0x2)
// CHECK-NEXT:    AuxSymbolCount: 0
// CHECK-NEXT:  }
