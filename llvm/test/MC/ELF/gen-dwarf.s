// RUN: llvm-mc -g -dwarf-version 2 -triple  i686-pc-linux-gnu %s -filetype=obj -o - | llvm-readobj -r | FileCheck %s
// RUN: not llvm-mc -g -dwarf-version 1  -triple  i686-pc-linux-gnu %s -filetype=asm -o - 2>&1 | FileCheck --check-prefix=DWARF1 %s
// RUN: llvm-mc -g -dwarf-version 2 -triple  i686-pc-linux-gnu %s -filetype=asm -o - | FileCheck --check-prefix=ASM --check-prefix=DWARF2 %s
// RUN: llvm-mc -g -dwarf-version 3 -triple  i686-pc-linux-gnu %s -filetype=asm -o - | FileCheck --check-prefix=ASM --check-prefix=DWARF3 %s
// RUN: llvm-mc -g -triple  i686-pc-linux-gnu %s -filetype=asm -o - | FileCheck --check-prefix=ASM --check-prefix=DWARF4 %s
// RUN: llvm-mc -g -dwarf-version 5  -triple  i686-pc-linux-gnu %s -filetype=asm -o - 2>&1 | FileCheck --check-prefix=DWARF5 %s
// RUN: not llvm-mc -g -dwarf-version 6  -triple  i686-pc-linux-gnu %s -filetype=asm -o - 2>&1 | FileCheck --check-prefix=DWARF6 %s


// Test that on ELF:
// 1. the debug info has a relocation to debug_abbrev and one to to debug_line.
// 2. the debug_aranges has relocations to text and debug_line.


    .text
    .globl foo
    .type foo, @function
    .align 4
foo:
    ret
    .size foo, .-foo

// CHECK:      Relocations [
// CHECK:        Section ({{[^ ]+}}) .rel.debug_info {
// CHECK-NEXT:     0x6 R_386_32 .debug_abbrev 0x0
// CHECK-NEXT:     0xC R_386_32 .debug_line 0x0
// CHECK:        }
// CHECK-NEXT:   Section ({{[^ ]+}}) .rel.debug_aranges {
// CHECK-NEXT:     0x6 R_386_32 .debug_info 0x0
// CHECK-NEXT:     0x10 R_386_32 .text 0x0
// CHECK-NEXT:   }
// CHECK:      ]

// First instance of the section is just to give it a label for debug_aranges to refer to
// ASM: .section .debug_info

// ASM: .section .debug_abbrev
// ASM-NEXT: [[ABBREV_LABEL:.Ltmp[0-9]+]]
// DWARF5: .section .debug_abbrev
// DWARF5-NEXT: [[ABBREV_LABEL:.Ltmp[0-9]+]]

// Second instance of the section has the CU
// ASM: .section .debug_info
// Dwarf version
// DWARF2: .short 2
// DWARF3: .short 3
// DWARF4: .short 4
// ASM-NEXT: .long [[ABBREV_LABEL]]
// DWARF5: .short 5
// DWARF5-NEXT: .byte 1
// DWARF5-NEXT: .byte 4
// DWARF5-NEXT: .long [[ABBREV_LABEL]]

// First .byte 1 is the abbreviation number for the compile_unit abbrev
// ASM: .byte 1
// ASM-NEXT: .long [[LINE_LABEL:.L[a-z0-9]+]]

// ASM: .section .debug_line
// ASM-NEXT: [[LINE_LABEL]]

// DWARF1: Dwarf version 1 is not supported.
// DWARF6: Dwarf version 6 is not supported.
