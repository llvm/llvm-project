// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj --symbols - | FileCheck %s

.zerofill __DATA,__bss,_fill0,1,0
.zerofill __DATA,__bss,_a,4,2
.zerofill __DATA,__bss,_fill1,1,0
.zerofill __DATA,__bss,_b,4,3
.zerofill __DATA,__bss,_fill2,1,0
.zerofill __DATA,__bss,_c,4,4
.zerofill __DATA,__bss,_fill3,1,0
.zerofill __DATA,__bss,_d,4,5

// CHECK: File: <stdin>
// CHECK: Format: Mach-O 32-bit i386
// CHECK: Arch: i386
// CHECK: AddressSize: 32bit
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name: _fill0 (34)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __bss (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x0
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _a (10)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __bss (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x4
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _fill1 (27)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __bss (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x8
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _b (7)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __bss (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x10
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _fill2 (20)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __bss (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x14
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _c (4)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __bss (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x20
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _fill3 (13)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __bss (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x24
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: _d (1)
// CHECK:     Type: Section (0xE)
// CHECK:     Section: __bss (0x2)
// CHECK:     RefType: UndefinedNonLazy (0x0)
// CHECK:     Flags [ (0x0)
// CHECK:     ]
// CHECK:     Value: 0x40
// CHECK:   }
// CHECK: ]
