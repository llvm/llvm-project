// RUN: llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -fdebug-compilation-dir=/tmp
// RUN: llvm-dwarfdump -a %t | FileCheck -check-prefix DWARF %s
// RUN: llvm-objdump -r %t | FileCheck -check-prefix RELOC %s

  .section foo, "ax"
b:
  mov r1, r1

// DWARF: .debug_abbrev contents:
// DWARF: Abbrev table for offset: 0x00000000
// DWARF: [1] DW_TAG_compile_unit DW_CHILDREN_yes
// DWARF:         DW_AT_stmt_list DW_FORM_sec_offset
// DWARF:         DW_AT_low_pc    DW_FORM_addr
// DWARF:         DW_AT_high_pc   DW_FORM_addr
// DWARF:         DW_AT_name      DW_FORM_string
// DWARF:         DW_AT_comp_dir  DW_FORM_string
// DWARF:         DW_AT_producer  DW_FORM_string
// DWARF:         DW_AT_language  DW_FORM_data2

// DWARF: .debug_info contents:
// DWARF: DW_TAG_compile_unit
// DWARF-NOT:         DW_TAG_
// DWARF:               DW_AT_low_pc (0x0000000000000000)
// DWARF:               DW_AT_high_pc (0x0000000000000004)

// DWARF: DW_TAG_label
// DWARF-NEXT: DW_AT_name ("b")


// DWARF: .debug_aranges contents:
// DWARF-NEXT: Address Range Header: length = 0x0000001c, version = 0x0002, cu_offset = 0x00000000, addr_size = 0x04, seg_size = 0x00
// DWARF-NEXT: [0x00000000, 0x00000004)


// DWARF: .debug_line contents:
// DWARF:      0x0000000000000000      7      0      1   0   0  is_stmt
// DWARF-NEXT: 0x0000000000000004      7      0      1   0   0  is_stmt end_sequence


// DWARF-NOT: .debug_ranges contents:
// DWARF-NOT: .debug_pubnames contents:



// RELOC: RELOCATION RECORDS FOR [.rel.debug_info]:
// RELOC-NEXT: 00000006 R_ARM_ABS32 .debug_abbrev
// RELOC-NEXT: 0000000c R_ARM_ABS32 .debug_line
// RELOC-NEXT: R_ARM_ABS32 foo
// RELOC-NEXT: R_ARM_ABS32 foo
// RELOC-NEXT: R_ARM_ABS32 foo

// RELOC-NOT: RELOCATION RECORDS FOR [.rel.debug_ranges]:

// RELOC: RELOCATION RECORDS FOR [.rel.debug_aranges]:
// RELOC-NEXT: 00000006 R_ARM_ABS32 .debug_info
// RELOC-NEXT: 00000010 R_ARM_ABS32 foo
