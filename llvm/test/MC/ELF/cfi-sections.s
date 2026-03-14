// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -dwarf-version 2 %s -o - | llvm-readobj -S --sd - | FileCheck -check-prefix=ELF_64 -check-prefix=ELF_64_DWARF_2 %s
// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu -dwarf-version 2 %s -o - | llvm-readobj -S --sd - | FileCheck -check-prefix=ELF_32 -check-prefix=ELF_32_DWARF_2 %s

// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -dwarf-version 3 %s -o - | llvm-readobj -S --sd - | FileCheck -check-prefix=ELF_64 -check-prefix=ELF_64_DWARF_3 %s
// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu -dwarf-version 3 %s -o - | llvm-readobj -S --sd - | FileCheck -check-prefix=ELF_32 -check-prefix=ELF_32_DWARF_3 %s

// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -dwarf-version 4 %s -o - | llvm-readobj -S --sd - | FileCheck -check-prefix=ELF_64 -check-prefix=ELF_64_DWARF_4 %s
// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu -dwarf-version 4 %s -o - | llvm-readobj -S --sd - | FileCheck -check-prefix=ELF_32 -check-prefix=ELF_32_DWARF_4 %s

.cfi_sections .debug_frame

f1:
        .cfi_startproc
        nop
        .cfi_endproc

f2:
        .cfi_startproc
        nop
        .cfi_endproc

// ELF_64:        Section {
// ELF_64:          Name: .debug_frame
// ELF_64-NEXT:     Type: SHT_PROGBITS
// ELF_64-NEXT:     Flags [
// ELF_64-NEXT:     ]
// ELF_64-NEXT:     Address: 0x0
// ELF_64-NEXT:     Offset: 0x48
// ELF_64-NEXT:     Size: 72
// ELF_64-NEXT:     Link: 0
// ELF_64-NEXT:     Info: 0
// ELF_64-NEXT:     AddressAlignment: 8
// ELF_64-NEXT:     EntrySize: 0
// ELF_64-NEXT:     SectionData (
// ELF_64_DWARF_2-NEXT:       0000: 14000000 FFFFFFFF 01000178 100C0708
// ELF_64_DWARF_2-NEXT:       0010: 90010000 00000000 14000000 00000000
// ELF_64_DWARF_3-NEXT:       0000: 14000000 FFFFFFFF 03000178 100C0708
// ELF_64_DWARF_3-NEXT:       0010: 90010000 00000000 14000000 00000000
// ELF_64_DWARF_4-NEXT:       0000: 14000000 FFFFFFFF 04000800 0178100C
// ELF_64_DWARF_4-NEXT:       0010: 07089001 00000000 14000000 00000000
// ELF_64-NEXT:       0020: 00000000 00000000 01000000 00000000
// ELF_64-NEXT:       0030: 14000000 00000000 00000000 00000000
// ELF_64-NEXT:       0040: 01000000 00000000
// ELF_64-NEXT:     )
// ELF_64-NEXT:   }

// ELF_32:        Section {
// ELF_32:          Name: .debug_frame
// ELF_32-NEXT:     Type: SHT_PROGBITS
// ELF_32-NEXT:     Flags [
// ELF_32-NEXT:     ]
// ELF_32-NEXT:     Address: 0x0
// ELF_32-NEXT:     Offset: 0x38
// ELF_32-NEXT:     Size: 52
// ELF_32-NEXT:     Link: 0
// ELF_32-NEXT:     Info: 0
// ELF_32-NEXT:     AddressAlignment: 4
// ELF_32-NEXT:     EntrySize: 0
// ELF_32-NEXT:     SectionData (
// ELF_32_DWARF_2-NEXT:       0000: 10000000 FFFFFFFF 0100017C 080C0404
// ELF_32_DWARF_2-NEXT:       0010: 88010000 0C000000 00000000 00000000
// ELF_32_DWARF_3-NEXT:       0000: 10000000 FFFFFFFF 0300017C 080C0404
// ELF_32_DWARF_3-NEXT:       0010: 88010000 0C000000 00000000 00000000
// ELF_32_DWARF_4-NEXT:       0000: 10000000 FFFFFFFF 04000400 017C080C
// ELF_32_DWARF_4-NEXT:       0010: 04048801 0C000000 00000000 00000000
// ELF_32-NEXT:       0020: 01000000 0C000000 00000000 01000000
// ELF_32-NEXT:       0030: 01000000

// ELF_32-NEXT:     )
// ELF_32-NEXT:   }
