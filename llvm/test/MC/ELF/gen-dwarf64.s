## This checks that llvm-mc is able to produce 64-bit debug info.

# RUN: llvm-mc -g -dwarf-version 5 -dwarf64 -triple x86_64 %s -filetype=obj -o %t5.o
# RUN: llvm-readobj -r %t5.o | FileCheck --check-prefixes=REL,REL5 %s
# RUN: llvm-dwarfdump -v %t5.o | FileCheck --check-prefixes=DUMP,DUMP5 %s

## The references to other debug info sections are 64-bit, as required for DWARF64.
# REL:         Section ({{[0-9]+}}) .rela.debug_info {
# REL-NEXT:      R_X86_64_64 .debug_abbrev 0x0
# REL-NEXT:      R_X86_64_64 .debug_line 0x0
# REL5:        Section ({{[0-9]+}}) .rela.debug_line {
# REL5-NEXT:     R_X86_64_64 .debug_line_str 0x0
# REL5-NEXT:     R_X86_64_64 .debug_line_str 0x

# DUMP:       .debug_info contents:
# DUMP-NEXT:  0x00000000: Compile Unit: {{.*}} format = DWARF64
# DUMP:       DW_TAG_compile_unit [1] *
# DUMP5-NEXT:   DW_AT_stmt_list [DW_FORM_sec_offset] (0x0000000000000000)
# DUMP:       DW_TAG_label [2]
# DUMP-NEXT:    DW_AT_name [DW_FORM_string] ("foo")

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

    .section .foo, "ax", @progbits
foo:
    nop
