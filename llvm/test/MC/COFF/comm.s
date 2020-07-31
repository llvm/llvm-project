// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | llvm-readobj --symbols - | FileCheck %s

.lcomm _a,4,4
.comm	_b, 4, 2
// _c has size 1 but align 32, the value field is the max of size and align.
.comm	_c, 1, 5


// CHECK:       Symbol {
// CHECK:         Name: _a
// CHECK-NEXT:    Value:
// CHECK-NEXT:    Section: .bss
// CHECK-NEXT:    BaseType: Null
// CHECK-NEXT:    ComplexType: Null
// CHECK-NEXT:    StorageClass: Static
// CHECK-NEXT:    AuxSymbolCount: 0
// CHECK-NEXT:  }

// CHECK:       Symbol {
// CHECK:         Name: _b
// CHECK-NEXT:    Value: 4
// CHECK-NEXT:    Section:  IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:    BaseType: Null
// CHECK-NEXT:    ComplexType: Null
// CHECK-NEXT:    StorageClass: External
// CHECK-NEXT:    AuxSymbolCount: 0
// CHECK-NEXT:  }

// CHECK:       Symbol {
// CHECK:         Name: _c
// CHECK-NEXT:    Value: 32
// CHECK-NEXT:    Section:  IMAGE_SYM_UNDEFINED (0)
// CHECK-NEXT:    BaseType: Null
// CHECK-NEXT:    ComplexType: Null
// CHECK-NEXT:    StorageClass: External
// CHECK-NEXT:    AuxSymbolCount: 0
// CHECK-NEXT:  }
