// RUN: llvm-mc -dwarf-version=2 -g -triple aarch64-apple-darwin10 %s -filetype=obj -o - \
// RUN:     | llvm-dwarfdump --debug-abbrev --debug-info - | FileCheck %s --check-prefix=DWARF2

// RUN: llvm-mc -dwarf-version=4 -g -triple aarch64-apple-darwin10 %s -filetype=obj -o - \
// RUN:     | llvm-dwarfdump --debug-abbrev --debug-info - | FileCheck %s --check-prefix=DWARF4

// RUN: llvm-mc -dwarf-version=6 -g -triple aarch64-apple-darwin10 %s -filetype=obj -o - \
// RUN:     | llvm-dwarfdump --debug-abbrev --debug-info - | FileCheck %s --check-prefix=DWARF6

_main:
	nop

// DWARF2:  DW_AT_language DW_FORM_data2
// DWARF2:  DW_AT_language (DW_LANG_Mips_Assembler)

// DWARF4:  DW_AT_language DW_FORM_data2
// DWARF4:  DW_AT_language (DW_LANG_Mips_Assembler)

// DWARF6:  DW_AT_language_name DW_FORM_data2
// DWARF6:  DW_AT_language_name (DW_LNAME_Assembly)
