# Test object to verify dwarfdump handles a DWARF-64 v5 line header.
# FIXME: Make the other headers DWARF-64 also.
# FIXME: Add variants for earlier DWARF versions.

# RUN: llvm-mc -triple x86_64-unknown-linux --defsym ELF=0 \
# RUN:     -filetype=obj -o - %s | llvm-dwarfdump -v - | FileCheck %s

# RUN: llvm-mc -triple x86_64-apple-darwin --defsym MACHO=0 \
# RUN:     -filetype=obj -o - %s | llvm-dwarfdump -v - | FileCheck %s

.ifdef ELF
        .section .debug_str,"MS",@progbits,1
.endif
.ifdef MACHO
        .section __DWARF,__debug_str,regular,debug
.endif
str_producer:
        .asciz  "Handmade DWARF producer"
str_CU_5:
        .asciz  "V5_compile_unit"
str_LT_5a:
        .asciz  "Directory5a"
str_LT_5b:
        .asciz  "Directory5b"

.ifdef ELF
        .section .debug_abbrev,"",@progbits
.endif
.ifdef MACHO
        .section __DWARF,__debug_abbrev,regular,debug
.endif
abbrev:
        .byte   0x01    # Abbrev code
        .byte   0x11    # DW_TAG_compile_unit
        .byte   0x00    # DW_CHILDREN_no
        .byte   0x25    # DW_AT_producer
        .byte   0x0e    # DW_FORM_strp
        .byte   0x03    # DW_AT_name
        .byte   0x0e    # DW_FORM_strp
        .byte   0x10    # DW_AT_stmt_list
        .byte   0x17    # DW_FORM_sec_offset
        .byte   0x00    # EOM(1)
        .byte   0x00    # EOM(2)
        .byte   0x00    # EOM(3)

.ifdef ELF
        .section .debug_info,"",@progbits
.endif
.ifdef MACHO
        .section __DWARF,__debug_info,regular,debug
.endif

# DWARF-32 v5 normal CU header.
Lset0 = CU_5_end-CU_5_version   # Length of Unit
        .long   Lset0
CU_5_version:
        .short  5               # DWARF version number
        .byte   1               # DWARF Unit Type
        .byte   8               # Address Size (in bytes)
.ifdef ELF
        .long   abbrev          # Offset Into Abbrev. Section
.endif
.ifdef MACHO
        .long   0
.endif
# The compile-unit DIE, with DW_AT_producer, DW_AT_name, DW_AT_stmt_list.
        .byte   1
        .long   str_producer
        .long   str_CU_5
.ifdef ELF
        .long   LH_5_start
.endif
.ifdef MACHO
        .long   0
.endif
        .byte   0 # NULL
CU_5_end:

# CHECK-LABEL: .debug_info contents:
# CHECK: 0x00000000: Compile Unit: length = 0x00000016, format = DWARF32, version = 0x0005, unit_type = DW_UT_compile, abbr_offset = 0x0000, addr_size = 0x08 (next unit at 0x0000001a)
# CHECK: 0x0000000c: DW_TAG_compile_unit
# CHECK-NEXT: DW_AT_producer {{.*}} "Handmade DWARF producer"
# CHECK-NEXT: DW_AT_name {{.*}} "V5_compile_unit"
# CHECK-NEXT: DW_AT_stmt_list {{.*}} (0x00000000)

.ifdef ELF
        .section .debug_line,"",@progbits
.endif
.ifdef MACHO
        .section __DWARF,__debug_line,regular,debug
.endif

# DWARF-64 v5 line-table header.
LH_5_start:
        .long   -1
Lset1 = LH_5_end-LH_5_version   # Length of Unit
        .quad   Lset1
LH_5_version:
        .short  5               # DWARF version number
        .byte   8               # Address Size
        .byte   0               # Segment Selector Size
Lset2 = LH_5_header_end-LH_5_params     # Length of Prologue
        .quad   Lset2
LH_5_params:
        .byte   1               # Minimum Instruction Length
        .byte   1               # Maximum Operations per Instruction
        .byte   1               # Default is_stmt
        .byte   -5              # Line Base
        .byte   14              # Line Range
        .byte   13              # Opcode Base
        .byte   0               # Standard Opcode Lengths
        .byte   1
        .byte   1
        .byte   1
        .byte   1
        .byte   0
        .byte   0
        .byte   0
        .byte   1
        .byte   0
        .byte   0
        .byte   1
        # Directory table format
        .byte   1               # One element per directory entry
        .byte   1               # DW_LNCT_path
        .byte   0x0e            # DW_FORM_strp (-> .debug_str)
        # Directory table entries
        .byte   2               # Two directories
        .quad   str_LT_5a
        .quad   str_LT_5b
        # File table format
        .byte   4               # Four elements per file entry
        .byte   1               # DW_LNCT_path
        .byte   0x08            # DW_FORM_string
        .byte   2               # DW_LNCT_directory_index
        .byte   0x0b            # DW_FORM_data1
        .byte   3               # DW_LNCT_timestamp
        .byte   0x0f            # DW_FORM_udata
        .byte   4               # DW_LNCT_size
        .byte   0x0f            # DW_FORM_udata
        # File table entries
        .byte   2               # Two files
        .asciz "File5a"
        .byte   0
        .byte   0x51
        .byte   0x52
        .asciz "File5b"
        .byte   1
        .byte   0x53
        .byte   0x54
LH_5_header_end:
        # Line number program, which is empty.
LH_5_end:

# CHECK-LABEL: .debug_line contents:
# CHECK: Line table prologue:
# CHECK: total_length: 0x0000000000000050
# CHECK: version: 5
# CHECK: address_size: 8
# CHECK: seg_select_size: 0
# CHECK: prologue_length: 0x0000000000000044
# CHECK: max_ops_per_inst: 1
# CHECK: include_directories[  0] = .debug_str[0x0000000000000028] = "Directory5a"
# CHECK: include_directories[  1] = .debug_str[0x0000000000000034] = "Directory5b"
# CHECK-NOT: include_directories
# CHECK: file_names[  0]:
# CHECK-NEXT: name: "File5a"
# CHECK-NEXT: dir_index: 0
# CHECK-NEXT: mod_time: 0x00000051
# CHECK-NEXT: length: 0x00000052
# CHECK: file_names[  1]:
# CHECK-NEXT: name: "File5b"
# CHECK-NEXT: dir_index: 1
# CHECK-NEXT: mod_time: 0x00000053
# CHECK-NEXT: length: 0x00000054
# CHECK-NOT: file_names
