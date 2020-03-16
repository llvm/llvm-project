// RUN: llvm-mc < %s -triple=avr -filetype=obj -o %t -g -fdebug-compilation-dir=/tmp
// RUN: llvm-dwarfdump -v %t | FileCheck -check-prefix DWARF %s
// RUN: llvm-objdump -r %t | FileCheck --check-prefix=RELOC %s

// If there is no code in an assembly file, no debug info is produced

.section .data, "aw"
a:
.long 42

// DWARF: elf32-avr
// DWARF-NOT: contents:
// DWARF: .debug_line contents:

// RELOC-NOT: RELOCATION RECORDS FOR [.rel.debug_info]:

// RELOC-NOT: RELOCATION RECORDS FOR [.rel.debug_ranges]:

// RELOC-NOT: RELOCATION RECORDS FOR [.rel.debug_aranges]:
