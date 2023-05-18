## This tests if .debug_loclists.dwo are correctly read if they are part
## of a dwp file.

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s --defsym MAIN=0 > %t
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t.dwp
# RUN: %lldb %t -o "image lookup -v -s lookup_loclists" -o exit | FileCheck %s

# CHECK-LABEL: image lookup -v -s lookup_loclists
# CHECK: Variable: id = {{.*}}, name = "x0", type = "int", valid ranges = <block>, location = [0x0000000000000000, 0x0000000000000003) -> DW_OP_reg0 RAX,
# CHECK: Variable: id = {{.*}}, name = "x1", type = "int", valid ranges = <block>, location = [0x0000000000000002, 0x0000000000000004) -> DW_OP_reg1 RDX,

## This part is kept in both the main and the dwp file to be able to reference the offsets.
loclists:
        nop
        nop
.Ltmp1:
lookup_loclists:
        nop
.Ltmp2:
        nop
.Ltmp3:
        nop
.Lloclists_end:

## The main file.
.ifdef MAIN
        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   74                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .byte   0x76                    # DW_AT_dwo_name
        .byte   8                       # DW_FORM_string
        .byte   115                     # DW_AT_addr_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   85                      # DW_AT_ranges
        .byte   35                      # DW_FORM_rnglistx
        .byte   116                     # DW_AT_rnglists_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   4                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .quad   1026699901672188186      # DWO id
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .asciz  "debug_loclists-dwp.dwo"  # DW_AT_dwo_name
        .long   .Laddr_table_base0      # DW_AT_addr_base
        .quad   loclists                # DW_AT_low_pc
        .byte   0                       # DW_AT_ranges
        .long   .Lskel_rnglists_table_base # DW_AT_rnglists_base
.Ldebug_info_end0:
        .section        .debug_rnglists,"",@progbits
        .long   .Lskel_rnglist_table_end-.Lskel_rnglist_table_start # Length
.Lskel_rnglist_table_start:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   1                       # Offset entry count
.Lskel_rnglists_table_base:
        .long   .Lskel_ranges0-.Lskel_rnglists_table_base
.Lskel_ranges0:
        .byte   7                       # DW_RLE_start_length
        .quad   loclists
        .uleb128   .Lloclists_end-loclists
        .byte   0                       # DW_RLE_end_of_list
.Lskel_rnglist_table_end:
        .section        .debug_addr,"",@progbits
        .long   .Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
        .short  5                       # DWARF version number
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
.Laddr_table_base0:
        .quad   loclists
        .quad   .Ltmp1
.Ldebug_addr_end0:

.else
## DWP file starts here.

        .section        .debug_loclists.dwo,"e",@progbits
## Start the section with an unused table to check that the reading offset
## of the real table is correctly adjusted.
        .long .LLLDummyEnd-.LLLDummyVersion # Length of Unit
.LLLDummyVersion:
        .short 5                            # Version
        .byte 8                             # Address size
        .byte 0                             # Segment selector size
        .long 0                             # Offset entry count
        .byte 0                             # DW_LLE_end_of_list
.LLLDummyEnd:
.LLLBegin:
        .long   .Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0 # Length
.Ldebug_loclist_table_start0:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   2                       # Offset entry count
.Lloclists_table_base:
        .long   .Ldebug_loc0-.Lloclists_table_base
        .long   .Ldebug_loc1-.Lloclists_table_base
.Ldebug_loc0:
        .byte   4                       # DW_LLE_offset_pair
        .uleb128 loclists-loclists
        .uleb128  .Ltmp2-loclists
        .uleb128 1                      # Expression size
        .byte   80                      # super-register DW_OP_reg0
        .byte   0                       # DW_LLE_end_of_list
.Ldebug_loc1:
        .byte   3                       # DW_LLE_startx_length
        .uleb128 1
        .uleb128  .Ltmp3-.Ltmp1
        .uleb128 1                      # Expression size
        .byte   81                      # super-register DW_OP_reg1
        .byte   0                       # DW_LLE_end_of_list
.Ldebug_loclist_table_end0:
.LLLEnd:
        .section        .debug_abbrev.dwo,"e",@progbits
.LAbbrevBegin:
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .byte   27                      # DW_FORM_addrx
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   5                       # DW_TAG_formal_parameter
        .byte   0                       # DW_CHILDREN_no
        .byte   2                       # DW_AT_location
        .byte   0x22                    # DW_FORM_loclistx
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
.LAbbrevEnd:
        .section        .debug_info.dwo,"e",@progbits
.LCUBegin:
.Lcu_begin1:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  5                       # DWARF version number
        .byte   5                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   0                       # Offset Into Abbrev. Section
        .quad   1026699901672188186      # DWO id
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .short  12                      # DW_AT_language
        .byte   2                       # Abbrev [2] DW_TAG_subprogram
        .byte   0                       # DW_AT_low_pc
        .long   .Lloclists_end-loclists # DW_AT_high_pc
        .asciz  "loclists"              # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .uleb128 0                      # DW_AT_location
        .asciz  "x0"                    # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .uleb128 1                      # DW_AT_location
        .asciz  "x1"                    # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   0                       # End Of Children Mark
.Lint:
        .byte   4                       # Abbrev [4] DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end1:
.LCUEnd:
        .section .debug_cu_index, "", @progbits
## Header:
        .short 5                        # Version
        .short 0                        # Padding
        .long 3                         # Section count
        .long 1                         # Unit count
        .long 2                         # Slot count
## Hash Table of Signatures:
        .quad 1026699901672188186
        .quad 0
## Parallel Table of Indexes:
        .long 1
        .long 0
## Table of Section Offsets:
## Row 0:
        .long 1                         # DW_SECT_INFO
        .long 3                         # DW_SECT_ABBREV
        .long 5                         # DW_SECT_LOCLISTS
## Row 1:
        .long 0                         # Offset in .debug_info.dwo
        .long 0                         # Offset in .debug_abbrev.dwo
        .long .LLLBegin-.debug_loclists.dwo # Offset in .debug_loclists.dwo
## Table of Section Sizes:
        .long .LCUEnd-.LCUBegin         # Size in .debug_info.dwo
        .long .LAbbrevEnd-.LAbbrevBegin # Size in .debug_abbrev.dwo
        .long .LLLEnd-.LLLBegin         # Size in .debug_loclists.dwo
.endif
