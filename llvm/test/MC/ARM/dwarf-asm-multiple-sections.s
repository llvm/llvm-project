// RUN: llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -fdebug-compilation-dir=/tmp
// RUN: llvm-dwarfdump %t | FileCheck -check-prefix DWARF %s
// RUN: llvm-objdump -r %t | FileCheck -check-prefix RELOC %s
// RUN: llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -dwarf-version 2 2>&1 | FileCheck -check-prefix VERSION %s
// RUN: not llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -dwarf-version 1 2>&1 | FileCheck -check-prefix DWARF1 %s
// RUN: not llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -dwarf-version 5 2>&1 | FileCheck -check-prefix DWARF5 %s
  .section .text, "ax"
a:
  mov r0, r0

  .section foo, "ax"
b:
  mov r1, r1

// DWARF: .debug_abbrev contents:
// DWARF: Abbrev table for offset: 0x00000000
// DWARF: [1] DW_TAG_compile_unit DW_CHILDREN_yes
// DWARF:         DW_AT_stmt_list DW_FORM_data4
// DWARF:         DW_AT_ranges    DW_FORM_data4
// DWARF:         DW_AT_name      DW_FORM_string
// DWARF:         DW_AT_comp_dir  DW_FORM_string
// DWARF:         DW_AT_producer  DW_FORM_string
// DWARF:         DW_AT_language  DW_FORM_data2

// DWARF: .debug_info contents:
// DWARF: 0x{{[0-9a-f]+}}: DW_TAG_compile_unit [1]
// CHECK-NOT-DWARF: DW_TAG_
// DWARF: DW_AT_ranges [DW_FORM_data4]      (0x00000000

// DWARF: 0x{{[0-9a-f]+}}:   DW_TAG_label [2] *
// DWARF-NEXT: DW_AT_name [DW_FORM_string]     ("a")

// DWARF: 0x{{[0-9a-f]+}}:   DW_TAG_label [2] *
// DWARF-NEXT: DW_AT_name [DW_FORM_string]     ("b")


// DWARF: .debug_aranges contents:
// DWARF-NEXT: Address Range Header: length = 0x00000024, version = 0x0002, cu_offset = 0x00000000, addr_size = 0x04, seg_size = 0x00
// DWARF-NEXT: [0x00000000 - 0x00000004)
// DWARF-NEXT: [0x00000000 - 0x00000004)


// DWARF: .debug_line contents:
// DWARF:      0x0000000000000000      9      0      1   0   0  is_stmt
// DWARF-NEXT: 0x0000000000000004      9      0      1   0   0  is_stmt end_sequence
// DWARF-NEXT: 0x0000000000000000     13      0      1   0   0  is_stmt
// DWARF-NEXT: 0x0000000000000004     13      0      1   0   0  is_stmt end_sequence


// DWARF: .debug_ranges contents:
// DWARF: 00000000 ffffffff 00000000
// DWARF: 00000000 00000000 00000004
// DWARF: 00000000 ffffffff 00000000
// DWARF: 00000000 00000000 00000004
// DWARF: 00000000 <End of list>



// RELOC: RELOCATION RECORDS FOR [.rel.debug_info]:
// RELOC-NEXT: 00000006 R_ARM_ABS32 .debug_abbrev
// RELOC-NEXT: 0000000c R_ARM_ABS32 .debug_line
// RELOC-NEXT: 00000010 R_ARM_ABS32 .debug_ranges
// RELOC-NEXT: R_ARM_ABS32 .text
// RELOC-NEXT: R_ARM_ABS32 foo

// RELOC: RELOCATION RECORDS FOR [.rel.debug_ranges]:
// RELOC-NEXT: 00000004 R_ARM_ABS32 .text
// RELOC-NEXT: 00000014 R_ARM_ABS32 foo

// RELOC: RELOCATION RECORDS FOR [.rel.debug_aranges]:
// RELOC-NEXT: 00000006 R_ARM_ABS32 .debug_info
// RELOC-NEXT: 00000010 R_ARM_ABS32 .text
// RELOC-NEXT: 00000018 R_ARM_ABS32 foo


// VERSION: {{.*}} warning: DWARF2 only supports one section per compilation unit

// DWARF1: Dwarf version 1 is not supported.
// DWARF5: Dwarf version 5 is not supported.
