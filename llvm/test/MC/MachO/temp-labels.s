// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -save-temp-labels -o - | llvm-readobj --symbols - | FileCheck %s

_f0:
        .long 0
L0:
        .long 0

// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _f0 (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: L0 (5)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __text (0x1)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x4
// CHECK:   }
// CHECK: ]
