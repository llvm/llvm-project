# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t
# RUN: llvm-dwarfdump %t | FileCheck %s


# CHECK:          DW_AT_location        (0x0000000c
# CHECK-NEXT:        Addr idx 0 (w/ length 1): DW_OP_reg0 RAX
# CHECK-NEXT:        [0x0000000000000001, 0x0000000000000002): DW_OP_reg1 RDX
# CHECK-NEXT:        [0x0000000000000002, 0x0000000000000003): DW_OP_reg2 RCX
# CHECK-NEXT:        [0x0000000000000003, 0x0000000000000004): DW_OP_reg3 RBX
# CHECK-NEXT:        Addr idx 57005 (w/ length 1): DW_OP_reg4 RSI)


        .text
f:                                      # @f
.Lf0:
        nop
.Lf1:
        nop
.Lf2:
        nop
.Lf3:
        nop
.Lf4:
.Lfend:
                                        # -- End function
        .section        .debug_loclists,"",@progbits
        .long   .Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0 # Length
.Ldebug_loclist_table_start0:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   0                       # Offset entry count
.Lloclists_table_base0:
.Ldebug_loc0:
        .byte   3                       # DW_LLE_startx_length
        .uleb128 0                      #   start idx
        .uleb128 .Lf1-.Lf0              #   length
        .byte   1                       # Loc expr size
        .byte   80                      # super-register DW_OP_reg0
        .byte   4                       # DW_LLE_offset_pair
        .uleb128 .Lf1-.Lf0              #   starting offset
        .uleb128 .Lf2-.Lf0              #   ending offset
        .byte   1                       # Loc expr size
        .byte   81                      # super-register DW_OP_reg1
        .byte   8                       # DW_LLE_start_length
        .quad   .Lf2                    #   starting offset
        .uleb128 .Lf3-.Lf2              #   length
        .byte   1                       # Loc expr size
        .byte   82                      # super-register DW_OP_reg2
        .byte   6                       # DW_LLE_base_address
        .quad   .Lf3                    #   base address
        .byte   4                       # DW_LLE_offset_pair
        .uleb128 .Lf3-.Lf3              #   starting offset
        .uleb128 .Lf4-.Lf3              #   ending offset
        .byte   1                       # Loc expr size
        .byte   83                      # super-register DW_OP_reg3
        .byte   3                       # DW_LLE_startx_length
        .uleb128 0xdead                 #   start idx
        .uleb128 .Lf1-.Lf0              #   length
        .byte   1                       # Loc expr size
        .byte   84                      # super-register DW_OP_reg4
        .byte   0                       # DW_LLE_end_of_list
.Ldebug_loclist_table_end0:

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   115                     # DW_AT_addr_base
        .byte   23                      # DW_FORM_sec_offset
        .ascii  "\214\001"              # DW_AT_loclists_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .byte   27                      # DW_FORM_addrx
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   5                       # DW_TAG_formal_parameter
        .byte   0                       # DW_CHILDREN_no
        .byte   2                       # DW_AT_location
        .byte   23                      # DW_FORM_sec_offset
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
        .byte   1                       # Abbrev [1] 0xc:0x3c DW_TAG_compile_unit
        .long   .Laddr_table_base0      # DW_AT_addr_base
        .long   .Lloclists_table_base0  # DW_AT_loclists_base
        .byte   2                       # Abbrev [2] 0x27:0x1c DW_TAG_subprogram
        .byte   0                       # DW_AT_low_pc
        .long   .Lfend-.Lf0             # DW_AT_high_pc
        .byte   3                       # Abbrev [3] 0x36:0xc DW_TAG_formal_parameter
        .long   .Ldebug_loc0            # DW_AT_location
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
        .quad   .Lf0
.Ldebug_addr_end0:
