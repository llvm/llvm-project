// RUN: llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -dwarf-version 5 -fdebug-compilation-dir=/tmp
// RUN: llvm-dwarfdump -v %t | FileCheck --check-prefixes=DWARF,DWARF5 %s
// RUN: llvm-dwarfdump --debug-line %t | FileCheck -check-prefix DWARF-DL -check-prefix DWARF-DL-5 -DDWVER=5 -DDWFILE=0 %s
// RUN: llvm-objdump -r %t | FileCheck --check-prefixes=RELOC,RELOC5 %s
// RUN: llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -dwarf-version 4 -fdebug-compilation-dir=/tmp
// RUN: llvm-dwarfdump -v %t | FileCheck -check-prefixes=DWARF,DWARF34,DWARF4 %s
// RUN: llvm-dwarfdump --debug-line %t | FileCheck --check-prefixes=DWARF-DL,DWARF-DL-3 -DDWVER=4 -DDWFILE=1 %s
// RUN: llvm-objdump -r %t | FileCheck --check-prefixes=RELOC,RELOC34,RELOC4 %s
// RUN: llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -dwarf-version 3 -fdebug-compilation-dir=/tmp
// RUN: llvm-dwarfdump -v %t | FileCheck --check-prefixes=DWARF,DWARF34,DWARF3 %s
// RUN: llvm-dwarfdump --debug-line %t | FileCheck --check-prefixes=DWARF-DL,DWARF-DL-3 -DDWVER=3 -DDWFILE=1 %s
// RUN: llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -dwarf-version 2 2>&1 | FileCheck -check-prefix VERSION %s
// RUN: not llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -dwarf-version 1 2>&1 | FileCheck -check-prefix DWARF1 %s
// RUN: not llvm-mc < %s -triple=armv7-linux-gnueabi -filetype=obj -o %t -g -dwarf-version 6 2>&1 | FileCheck -check-prefix DWARF6 %s
  .section .text, "ax"
a:
  mov r0, r0

  .section foo, "ax"
b:
  mov r1, r1

// Non-executable sections do not contribute address ranges.
  .section .rodata, "a"
c:
  mov r2, r2

// Non-SHF_ALLOC sections do not contribute address ranges.
  .section nonalloc, "x"
d:
  mov r3, r3

// DWARF: .debug_abbrev contents:
// DWARF: Abbrev table for offset: 0x00000000
// DWARF: [1] DW_TAG_compile_unit DW_CHILDREN_yes
// DWARF3:        DW_AT_stmt_list DW_FORM_data4
// DWARF4:        DW_AT_stmt_list DW_FORM_sec_offset
// DWARF5:        DW_AT_stmt_list DW_FORM_sec_offset
// DWARF3:        DW_AT_ranges    DW_FORM_data4
// DWARF4:        DW_AT_ranges    DW_FORM_sec_offset
// DWARF5:        DW_AT_ranges    DW_FORM_sec_offset
// DWARF:         DW_AT_name      DW_FORM_string
// DWARF:         DW_AT_comp_dir  DW_FORM_string
// DWARF:         DW_AT_producer  DW_FORM_string
// DWARF:         DW_AT_language  DW_FORM_data2

// DWARF: .debug_info contents:
// DWARF: 0x{{[0-9a-f]+}}: DW_TAG_compile_unit [1]
// DWARF-NOT: DW_TAG_
// DWARF3:  DW_AT_ranges [DW_FORM_data4]           (0x00000000
// DWARF4:  DW_AT_ranges [DW_FORM_sec_offset]      (0x00000000
// DWARF5:  DW_AT_ranges [DW_FORM_sec_offset]      (0x0000000c

// DWARF: 0x{{[0-9a-f]+}}:   DW_TAG_label [2]
// DWARF-NEXT: DW_AT_name [DW_FORM_string]     ("a")

// DWARF: 0x{{[0-9a-f]+}}:   DW_TAG_label [2]
// DWARF-NEXT: DW_AT_name [DW_FORM_string]     ("b")

// DWARF-NOT: DW_TAG_label


// DWARF: .debug_aranges contents:
// DWARF-NEXT: Address Range Header: length = 0x00000024, format = DWARF32, version = 0x0002, cu_offset = 0x00000000, addr_size = 0x04, seg_size = 0x00
// DWARF-NEXT: [0x00000000, 0x00000004)
// DWARF-NEXT: [0x00000000, 0x00000004)
// DWARF-EMPTY:


// DWARF-DL: .debug_line contents:
// DWARF-DL: version: [[DWVER]]
// DWARF-DL-5:    address_size: 4
// DWARF-DL-5:    include_directories[  0] = "/tmp"
// DWARF-DL:      file_names[  [[DWFILE]]]:
// DWARF-DL:      name: "{{(<stdin>|-)}}"
// DWARF-DL-5:      0x0000000000000000     17      0      0   0   0       0  is_stmt
// DWARF-DL-5-NEXT: 0x0000000000000004     17      0      0   0   0       0  is_stmt end_sequence
// DWARF-DL-5-NEXT: 0x0000000000000000     21      0      0   0   0       0  is_stmt
// DWARF-DL-5-NEXT: 0x0000000000000004     21      0      0   0   0       0  is_stmt end_sequence
// DWARF-DL-3:      0x0000000000000000     17      0      1   0   0       0  is_stmt
// DWARF-DL-3-NEXT: 0x0000000000000004     17      0      1   0   0       0  is_stmt end_sequence
// DWARF-DL-3-NEXT: 0x0000000000000000     21      0      1   0   0       0  is_stmt
// DWARF-DL-3-NEXT: 0x0000000000000004     21      0      1   0   0       0  is_stmt end_sequence
// DWARF-DL-EMPTY:


// DWARF34:      .debug_ranges contents:
// DWARF34-NEXT: 00000000 ffffffff 00000000
// DWARF34-NEXT: 00000000 00000000 00000004
// DWARF34-NEXT: 00000000 ffffffff 00000000
// DWARF34-NEXT: 00000000 00000000 00000004
// DWARF34-NEXT: 00000000 <End of list>

// DWARF5:      .debug_rnglists contents:
// DWARF5-NEXT: 0x00000000: range list header: length = 0x00000015, format = DWARF32, version = 0x0005, addr_size = 0x04, seg_size = 0x00, offset_entry_count = 0x00000000
// DWARF5-NEXT: ranges:
// DWARF5-NEXT: 0x0000000c: [DW_RLE_start_length]: 0x00000000, 0x00000004 => [0x00000000, 0x00000004)
// DWARF5-NEXT: 0x00000012: [DW_RLE_start_length]: 0x00000000, 0x00000004 => [0x00000000, 0x00000004)
// DWARF5-NEXT: 0x00000018: [DW_RLE_end_of_list ]


// Offsets are different in DWARF v5 due to different header layout.
// RELOC: RELOCATION RECORDS FOR [.debug_info]:
// RELOC4-NEXT: OFFSET TYPE VALUE
// RELOC4-NEXT: 00000006 R_ARM_ABS32 .debug_abbrev
// RELOC4-NEXT: 0000000c R_ARM_ABS32 .debug_line
// RELOC4-NEXT: 00000010 R_ARM_ABS32 .debug_ranges
// RELOC5-NEXT: OFFSET TYPE VALUE
// RELOC5-NEXT: 00000008 R_ARM_ABS32 .debug_abbrev
// RELOC5-NEXT: 0000000d R_ARM_ABS32 .debug_line
// RELOC5-NEXT: 00000011 R_ARM_ABS32 .debug_rnglists
// RELOC-NEXT: R_ARM_ABS32 .text
// RELOC-NEXT: R_ARM_ABS32 foo

// RELOC: RELOCATION RECORDS FOR [.debug_aranges]:
// RELOC-NEXT: OFFSET TYPE VALUE
// RELOC-NEXT: 00000006 R_ARM_ABS32 .debug_info
// RELOC-NEXT: 00000010 R_ARM_ABS32 .text
// RELOC-NEXT: 00000018 R_ARM_ABS32 foo

// RELOC34: RELOCATION RECORDS FOR [.debug_ranges]:
// RELOC34-NEXT: OFFSET TYPE VALUE
// RELOC34-NEXT: 00000004 R_ARM_ABS32 .text
// RELOC34-NEXT: 00000014 R_ARM_ABS32 foo

// RELOC5: RELOCATION RECORDS FOR [.debug_rnglists]:
// RELOC5-NEXT: OFFSET TYPE VALUE
// RELOC5-NEXT: 0000000d R_ARM_ABS32 .text
// RELOC5-NEXT: 00000013 R_ARM_ABS32 foo


// VERSION: {{.*}} warning: DWARF2 only supports one section per compilation unit

// DWARF1: Dwarf version 1 is not supported.
// DWARF6: Dwarf version 6 is not supported.
