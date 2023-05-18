// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --symbols - | FileCheck %s
//
// Check that the section itself is aligned.

        .byte 0
        
.zerofill __DATA,__bss,_a,1,0
.zerofill __DATA,__bss,_b,4,4

// CHECK: File: <stdin>
// CHECK: Format: Mach-O 32-bit i386
// CHECK: Arch: i386
// CHECK: AddressSize: 32bit
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _a (4)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __bss (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x10
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _b (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __bss (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x20
// CHECK:   }
// CHECK: ]
