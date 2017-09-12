// RUN: llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -fdebug-compilation-dir=/tmp
// RUN: llvm-dwarfdump -v %t | FileCheck -check-prefix DWARF %s
// RUN: llvm-objdump -r %t | FileCheck -check-prefix RELOC %s

// If there is no code in an assembly file, no debug info is produced

.section .data, "aw"
a:
.long 42

// DWARF: .debug_abbrev contents:
// DWARF-NEXT: < EMPTY >

// DWARF: .debug_info contents:

// DWARF: .debug_aranges contents:

// DWARF: .debug_line contents:

// DWARF: .debug_ranges contents:


// RELOC-NOT: RELOCATION RECORDS FOR [.rel.debug_info]:

// RELOC-NOT: RELOCATION RECORDS FOR [.rel.debug_ranges]:

// RELOC-NOT: RELOCATION RECORDS FOR [.rel.debug_aranges]:
