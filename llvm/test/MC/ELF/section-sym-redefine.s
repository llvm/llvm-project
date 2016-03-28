// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj  -t -r --expand-relocs | FileCheck %s

// Local symbol overriding section.
.section x1,"a",@progbits
.local  x1
.comm   x1,4,4
.long x1  // reloc: .bss + 0

// Section declared after local. Local symbol wins.
.local  x2
.comm   x2,4,4
.section x2,"a",@progbits
.long x2  // reloc: .bss + 4

// No overriding symbol.
.section x3,"a",@progbits
.long x3  // reloc: x3(section) + 0

// Global vs section.
.section x4,"a",@progbits
.long 0
.globl x4
.section foo, "a", @progbits
x4:
.long 0
.long x4  // reloc: x4(global) + 0

// Global vs implicit section
.globl .data
.data:
.long 42
.long .data // reloc: .data(global) + 0

// CHECK: Relocations [
// CHECK:   Section (4) .relax1 {
// CHECK:     Relocation {
// CHECK:       Offset: 0x0
// CHECK:       Type: R_X86_64_32 (10)
// CHECK:       Symbol: .bss (3)
// CHECK:       Addend: 0x0
// CHECK:     }
// CHECK:   }
// CHECK:   Section (7) .relax2 {
// CHECK:     Relocation {
// CHECK:       Offset: 0x0
// CHECK:       Type: R_X86_64_32 (10)
// CHECK:       Symbol: .bss (3)
// CHECK:       Addend: 0x4
// CHECK:     }
// CHECK:   }
// CHECK:   Section (9) .relax3 {
// CHECK:     Relocation {
// CHECK:       Offset: 0x0
// CHECK:       Type: R_X86_64_32 (10)
// CHECK:       Symbol: x3 (4)
// CHECK:       Addend: 0x0
// CHECK:     }
// CHECK:   }
// CHECK:   Section (12) .relafoo {
// CHECK:     Relocation {
// CHECK:       Offset: 0x4
// CHECK:       Type: R_X86_64_32 (10)
// CHECK:       Symbol: x4 (6)
// CHECK:       Addend: 0x0
// CHECK:     }
// CHECK:     Relocation {
// CHECK:       Offset: 0xC
// CHECK:       Type: R_X86_64_32 (10)
// CHECK:       Symbol: .data (5)
// CHECK:       Addend: 0x0
// CHECK:     }
// CHECK:   }
// CHECK: ]
// CHECK: Symbols [
// CHECK:   Symbol {
// CHECK:     Name:  (0)
// CHECK:     Value: 0x0
// CHECK:     Size: 0
// CHECK:     Binding: Local (0x0)
// CHECK:     Type: None (0x0)
// CHECK:     Other: 0
// CHECK:     Section: Undefined (0x0)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: x1 (67)
// CHECK:     Value: 0x0
// CHECK:     Size: 4
// CHECK:     Binding: Local (0x0)
// CHECK:     Type: Object (0x1)
// CHECK:     Other: 0
// CHECK:     Section: .bss (0x5)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: x2 (59)
// CHECK:     Value: 0x4
// CHECK:     Size: 4
// CHECK:     Binding: Local (0x0)
// CHECK:     Type: Object (0x1)
// CHECK:     Other: 0
// CHECK:     Section: .bss (0x5)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name:  (0)
// CHECK:     Value: 0x0
// CHECK:     Size: 0
// CHECK:     Binding: Local (0x0)
// CHECK:     Type: Section (0x3)
// CHECK:     Other: 0
// CHECK:     Section: .bss (0x5)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name:  (0)
// CHECK:     Value: 0x0
// CHECK:     Size: 0
// CHECK:     Binding: Local (0x0)
// CHECK:     Type: Section (0x3)
// CHECK:     Other: 0
// CHECK:     Section: x3 (0x8)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: .data (37)
// CHECK:     Value: 0x8
// CHECK:     Size: 0
// CHECK:     Binding: Global (0x1)
// CHECK:     Type: None (0x0)
// CHECK:     Other: 0
// CHECK:     Section: foo (0xB)
// CHECK:   }
// CHECK:   Symbol {
// CHECK:     Name: x4 (43)
// CHECK:     Value: 0x0
// CHECK:     Size: 0
// CHECK:     Binding: Global (0x1)
// CHECK:     Type: None (0x0)
// CHECK:     Other: 0
// CHECK:     Section: foo (0xB)
// CHECK:   }
// CHECK: ]
