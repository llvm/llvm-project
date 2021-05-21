# DW_AT_ranges can use DW_FORM_sec_offset (instead of DW_FORM_rnglistx).
# In such case DW_AT_rnglists_base does not need to be present.

# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "image lookup -v -s lookup_rnglists" \
# RUN:   -o exit | FileCheck %s

# Failure was the block range 1..2 was not printed plus:
# error: DW_AT_range-DW_FORM_sec_offset.s.tmp {0x0000003f}: DIE has DW_AT_ranges(0xc) attribute, but range extraction failed (missing or invalid range list table), please file a bug and attach the file at the start of this error message

# CHECK-LABEL: image lookup -v -s lookup_rnglists
# CHECK:  Function: id = {0x00000029}, name = "rnglists", range = [0x0000000000000000-0x0000000000000003)
# CHECK:    Blocks: id = {0x00000029}, range = [0x00000000-0x00000003)
# CHECK-NEXT:       id = {0x0000003f}, range = [0x00000001-0x00000002)

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj \
# RUN:   --defsym RNGLISTX=0 %s > %t-rnglistx
# RUN: %lldb %t-rnglistx -o "image lookup -v -s lookup_rnglists" \
# RUN:   -o exit 2>&1 | FileCheck --check-prefix=RNGLISTX %s

# RNGLISTX-LABEL: image lookup -v -s lookup_rnglists
# RNGLISTX: error: {{.*}} {0x0000003f}: DIE has DW_AT_ranges(DW_FORM_rnglistx 0x0) attribute, but range extraction failed (DW_FORM_rnglistx cannot be used without DW_AT_rnglists_base for CU at 0x00000000), please file a bug and attach the file at the start of this error message

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj \
# RUN:   --defsym RNGLISTX=0 --defsym RNGLISTBASE=0 %s > %t-rnglistbase
# RUN: %lldb %t-rnglistbase -o "image lookup -v -s lookup_rnglists" \
# RUN:   -o exit 2>&1 | FileCheck --check-prefix=RNGLISTBASE %s

# RNGLISTBASE-LABEL: image lookup -v -s lookup_rnglists
# RNGLISTBASE: error: {{.*}}-rnglistbase {0x00000043}: DIE has DW_AT_ranges(DW_FORM_rnglistx 0x0) attribute, but range extraction failed (invalid range list table index 0; OffsetEntryCount is 0, DW_AT_rnglists_base is 12), please file a bug and attach the file at the start of this error message

        .text
rnglists:
        nop
.Lblock1_begin:
lookup_rnglists:
        nop
.Lblock1_end:
        nop
.Lrnglists_end:

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   17                      # DW_AT_low_pc
        .byte   27                      # DW_FORM_addrx
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   115                     # DW_AT_addr_base
        .byte   23                      # DW_FORM_sec_offset
.ifdef RNGLISTBASE
        .byte   0x74                    # DW_AT_rnglists_base
        .byte   23                      # DW_FORM_sec_offset
.endif
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   11                      # DW_TAG_lexical_block
        .byte   0                       # DW_CHILDREN_no
        .byte   85                      # DW_AT_ranges
.ifndef RNGLISTX
        .byte   0x17                    # DW_FORM_sec_offset
.else
        .byte   0x23                    # DW_FORM_rnglistx
.endif
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   1                       # Abbrev [1] 0xc:0x5f DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .byte   0                       # DW_AT_low_pc
        .long   .Lrnglists_end-rnglists # DW_AT_high_pc
        .long   .Laddr_table_base0      # DW_AT_addr_base
.ifdef RNGLISTBASE
        .long   .Ldebug_ranges0         # DW_AT_rnglists_base
.endif
        .byte   2                       # Abbrev [2] 0x2b:0x37 DW_TAG_subprogram
        .quad   rnglists                # DW_AT_low_pc
        .long   .Lrnglists_end-rnglists # DW_AT_high_pc
        .asciz  "rnglists"              # DW_AT_name
        .byte   5                       # Abbrev [5] 0x52:0xf DW_TAG_lexical_block
.ifndef RNGLISTX
        .long   .Ldebug_ranges0         # DW_AT_ranges DW_FORM_sec_offset
.else
        .uleb128 0                      # DW_AT_ranges DW_FORM_rnglistx
.endif
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

        .section        .debug_addr,"",@progbits
        .long   .Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
        .short  5                       # DWARF version number
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
.Laddr_table_base0:
        .quad   rnglists
.Ldebug_addr_end0:

        .section        .debug_rnglists,"",@progbits
        .long   .Ldebug_rnglist_table_end0-.Ldebug_rnglist_table_start0 # Length
.Ldebug_rnglist_table_start0:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   0                       # Offset entry count
.Ldebug_ranges0:
        .byte   4                       # DW_RLE_offset_pair
        .uleb128 .Lblock1_begin-rnglists  #   starting offset
        .uleb128 .Lblock1_end-rnglists    #   ending offset
        .byte   0                       # DW_RLE_end_of_list
.Ldebug_rnglist_table_end0:
