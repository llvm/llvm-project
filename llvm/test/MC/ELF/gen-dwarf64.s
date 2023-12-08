## This checks that llvm-mc is able to produce 64-bit debug info.

# RUN: llvm-mc -g -dwarf-version 3 -dwarf64 -triple x86_64 %s -filetype=obj -o %t3.o
# RUN: llvm-readobj -r %t3.o | FileCheck --check-prefixes=REL,REL3 %s --implicit-check-not="R_{{.*}} .debug_"
# RUN: llvm-dwarfdump -v %t3.o | FileCheck --check-prefixes=DUMP,DUMP3 %s

# RUN: llvm-mc -g -dwarf-version 5 -dwarf64 -triple x86_64 %s -filetype=obj -o %t5.o
# RUN: llvm-readobj -r %t5.o | FileCheck --check-prefixes=REL,REL5 %s --implicit-check-not="R_{{.*}} .debug_"
# RUN: llvm-dwarfdump -v %t5.o | FileCheck --check-prefixes=DUMP,DUMP5 %s

## The references to other debug info sections are 64-bit, as required for DWARF64.
# REL:         Section ({{[0-9]+}}) .rela.debug_frame {
# REL-NEXT:      R_X86_64_64 .debug_frame 0x0
# REL:         Section ({{[0-9]+}}) .rela.debug_info {
# REL-NEXT:      R_X86_64_64 .debug_abbrev 0x0
# REL-NEXT:      R_X86_64_64 .debug_line 0x0
# REL3-NEXT:     R_X86_64_64 .debug_ranges 0x0
# REL5-NEXT:     R_X86_64_64 .debug_rnglists 0x14
# REL:         Section ({{[0-9]+}}) .rela.debug_aranges {
# REL-NEXT:      R_X86_64_64 .debug_info 0x0
# REL5:        Section ({{[0-9]+}}) .rela.debug_line {
# REL5-NEXT:     R_X86_64_64 .debug_line_str 0x0
# REL5-NEXT:     R_X86_64_64 .debug_line_str 0x

## DW_FORM_sec_offset was introduced in DWARFv4.
## For DWARFv3, DW_FORM_data8 is used instead.
# DUMP:       .debug_abbrev contents:
# DUMP3:        DW_AT_stmt_list DW_FORM_data8
# DUMP3-NEXT:   DW_AT_ranges    DW_FORM_data8
# DUMP5:        DW_AT_stmt_list DW_FORM_sec_offset
# DUMP5-NEXT:   DW_AT_ranges    DW_FORM_sec_offset

# DUMP:       .debug_info contents:
# DUMP-NEXT:  0x00000000: Compile Unit: {{.*}} format = DWARF64
# DUMP:       DW_TAG_compile_unit [1] *
# DUMP3-NEXT:   DW_AT_stmt_list [DW_FORM_data8] (0x0000000000000000)
# DUMP5-NEXT:   DW_AT_stmt_list [DW_FORM_sec_offset] (0x0000000000000000)
# DUMP3-NEXT:   DW_AT_ranges [DW_FORM_data8] (0x0000000000000000
# DUMP5-NEXT:   DW_AT_ranges [DW_FORM_sec_offset] (0x0000000000000014
# DUMP-NEXT:      [0x0000000000000000, 0x0000000000000001) ".foo"
# DUMP-NEXT:      [0x0000000000000000, 0x0000000000000001) ".bar")
# DUMP:       DW_TAG_label [2]
# DUMP-NEXT:    DW_AT_name [DW_FORM_string] ("foo")
# DUMP:       DW_TAG_label [2]
# DUMP-NEXT:    DW_AT_name [DW_FORM_string] ("bar")

# DUMP:       .debug_frame contents:
# DUMP:       00000000 {{([[:xdigit:]]{16})}} ffffffffffffffff CIE
# DUMP-NEXT:    Format: DWARF64
# DUMP:       {{([[:xdigit:]]{8})}} {{([[:xdigit:]]{16})}} 0000000000000000 FDE cie=00000000 pc=00000000...00000001
# DUMP-NEXT:    Format: DWARF64

## Even though the debug info sections are in the 64-bit format,
## .eh_frame is still generated as 32-bit.
# DUMP:       .eh_frame contents:
# DUMP:       00000000 {{([[:xdigit:]]{8})}} 00000000 CIE
# DUMP-NEXT:    Format: DWARF32
# DUMP:       {{([[:xdigit:]]{8})}} {{([[:xdigit:]]{8})}} {{([[:xdigit:]]{8})}} FDE cie=00000000 pc=00000000...00000001
# DUMP-NEXT:    Format: DWARF32

# DUMP:       .debug_aranges contents:
# DUMP-NEXT:  Address Range Header: length = 0x0000000000000044, format = DWARF64, version = 0x0002, cu_offset = 0x0000000000000000, addr_size = 0x08, seg_size = 0x00
# DUMP-NEXT:  [0x0000000000000000,  0x0000000000000001)
# DUMP-NEXT:  [0x0000000000000000,  0x0000000000000001)
# DUMP-EMPTY:

# DUMP:       .debug_line contents:
# DUMP-NEXT:  debug_line[0x00000000]
# DUMP-NEXT:  Line table prologue:
# DUMP-NEXT:      total_length:
# DUMP-NEXT:            format: DWARF64
# DUMP5:      include_directories[  0] = .debug_line_str[0x0000000000000000] = "[[DIR:.+]]"
# DUMP5-NEXT: file_names[  0]:
# DUMP5-NEXT:            name: .debug_line_str[0x00000000[[FILEOFF:[[:xdigit:]]{8}]]] = "[[FILE:.+]]"
# DUMP5-NEXT:       dir_index: 0

# DUMP5:      .debug_line_str contents:
# DUMP5-NEXT: 0x00000000: "[[DIR]]"
# DUMP5-NEXT: 0x[[FILEOFF]]: "[[FILE]]"

# DUMP3:      .debug_ranges contents:
# DUMP3-NEXT: 00000000 ffffffffffffffff 0000000000000000
# DUMP3-NEXT: 00000000 0000000000000000 0000000000000001
# DUMP3-NEXT: 00000000 ffffffffffffffff 0000000000000000
# DUMP3-NEXT: 00000000 0000000000000000 0000000000000001
# DUMP3-NEXT: 00000000 <End of list>

# DUMP5:      .debug_rnglists contents:
# DUMP5-NEXT: 0x00000000: range list header: length = 0x000000000000001d, format = DWARF64, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000
# DUMP5-NEXT: ranges:
# DUMP5-NEXT: 0x00000014: [DW_RLE_start_length]: 0x0000000000000000, 0x0000000000000001 => [0x0000000000000000, 0x0000000000000001)
# DUMP5-NEXT: 0x0000001e: [DW_RLE_start_length]: 0x0000000000000000, 0x0000000000000001 => [0x0000000000000000, 0x0000000000000001)
# DUMP5-NEXT: 0x00000028: [DW_RLE_end_of_list ]

    .cfi_sections .eh_frame, .debug_frame

    .section .foo, "ax", @progbits
foo:
    nop

    .section .bar, "ax", @progbits
bar:
    .cfi_startproc
    nop
    .cfi_endproc
