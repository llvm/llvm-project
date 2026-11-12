## This tests if .debug_rnglists.dwo are correctly read if they are part
## of a dwp file.

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s --defsym MAIN=0 > %t
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t.dwp
# RUN: %lldb %t -o "image lookup -v -s lookup_rnglists" -o exit | FileCheck %s

# CHECK-LABEL: image lookup -v -s lookup_rnglists
# CHECK:  Function: id = {{.*}}, name = "rnglists", range = [0x0000000000000000-0x0000000000000003)
# CHECK:    Blocks: id = {{.*}}, range = [0x00000000-0x00000003)
# CHECK-NEXT:       id = {{.*}}, range = [0x00000001-0x00000002)

        .text
rnglists:
        nop
.Lblock1_begin:
lookup_rnglists:
        nop
.Lblock1_end:
        nop
.Lrnglists_end:

## The main file.
.ifdef MAIN
        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .byte   0x76                    # DW_AT_dwo_name
        .byte   8                       # DW_FORM_string
        .byte   115                     # DW_AT_addr_base
        .byte   23                      # DW_FORM_sec_offset
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
        .quad   1026699901672188186     # DWO id
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .asciz  "debug_rnglists-dwp.s.tmp.dwo"  # DW_AT_dwo_name
        .long   .Laddr_table_base0      # DW_AT_addr_base
        .byte   0                       # DW_AT_ranges
        .long   .Lskel_rnglists_table_base # DW_AT_rnglists_base
.Ldebug_info_end0:

        .section        .debug_addr,"",@progbits
        .long   .Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
        .short  5                       # DWARF version number
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
.Laddr_table_base0:
        .quad   rnglists
        .quad   .Lblock1_begin
.Ldebug_addr_end0:

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
        .quad   rnglists
        .uleb128   .Lrnglists_end-rnglists
        .byte   0                       # DW_RLE_end_of_list
.Lskel_rnglist_table_end:
 .else
 ## DWP file starts here.
        .section        .debug_abbrev.dwo,"e",@progbits
.LAbbrevBegin:
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
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
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   11                      # DW_TAG_lexical_block
        .byte   0                       # DW_CHILDREN_no
        .byte   85                      # DW_AT_ranges
        .byte   35                      # DW_FORM_rnglistx
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
.LAbbrevEnd:
        .section        .debug_info.dwo,"e",@progbits
.LCUBegin:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  5                       # DWARF version number
        .byte   5                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   0                       # Offset Into Abbrev. Section
        .quad   1026699901672188186      # DWO id
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .byte   2                       # Abbrev [2] DW_TAG_subprogram
        .byte   0                       # DW_AT_low_pc
        .long   .Lrnglists_end-rnglists # DW_AT_high_pc
        .asciz  "rnglists"              # DW_AT_name
        .byte   5                       # Abbrev [5] DW_TAG_lexical_block
        .byte   0                       # DW_AT_ranges
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end1:
.LCUEnd:
        .section        .debug_rnglists.dwo,"e",@progbits
## Fake rnglists to check if the cu index is taken into account
        .long   .Lfake_rnglist_end-.Lfake_rnglist_start # Length
.Lfake_rnglist_start:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   0                       # Offset entry count
        .byte   0                       # DW_RLE_end_of_list
.Lfake_rnglist_end:
.LRLBegin:
        .long   .Ldwo_rnglist_table_end-.Ldwo_rnglist_table_start # Length
.Ldwo_rnglist_table_start:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   1                       # Offset entry count
.Ldwo_rnglists_table_base:
        .long   .Ldwo_ranges-.Ldwo_rnglists_table_base
.Ldwo_ranges:
        .byte   3                       # DW_RLE_startx_length
        .uleb128 1
        .uleb128 .Lblock1_end-.Lblock1_begin
        .byte   0                       # DW_RLE_end_of_list
.Ldwo_rnglist_table_end:
.LRLEnd:
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
        .long 8                         # DW_SECT_RNGLISTS
## Row 1:
        .long 0                         # Offset in .debug_info.dwo
        .long 0                         # Offset in .debug_abbrev.dwo
        .long .LRLBegin-.debug_rnglists.dwo # Offset in .debug_rnglists.dwo
## Table of Section Sizes:
        .long .LCUEnd-.LCUBegin         # Size in .debug_info.dwo
        .long .LAbbrevEnd-.LAbbrevBegin # Size in .debug_abbrev.dwo
        .long .LRLEnd-.LRLBegin         # Size in .debug_rnglists.dwo
.endif
