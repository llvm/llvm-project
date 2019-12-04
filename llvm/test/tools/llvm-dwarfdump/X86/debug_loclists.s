# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t
# RUN: llvm-dwarfdump -debug-info -debug-loclists %t \
# RUN:   | FileCheck %s --check-prefix=REGULAR --check-prefix=BOTH
# RUN: llvm-dwarfdump -debug-info -debug-loclists --verbose %t \
# RUN:   | FileCheck %s --check-prefix=VERBOSE --check-prefix=BOTH


# BOTH:          DW_AT_location {{.*}}(0x0000000c

# REGULAR-NEXT:      [0x0000000000000000, 0x0000000000000001): DW_OP_reg0 RAX
# VERBOSE-NEXT:      [0x0000000000000000, 0x0000000000000001) ".text": DW_OP_reg0 RAX

# REGULAR-NEXT:      [0x0000000000000001, 0x0000000000000002): DW_OP_reg1 RDX
# VERBOSE-NEXT:      [0x0000000000000001, 0x0000000000000002) ".text": DW_OP_reg1 RDX

# REGULAR-NEXT:      [0x0000000000000002, 0x0000000000000003): DW_OP_reg2 RCX
# VERBOSE-NEXT:      [0x0000000000000002, 0x0000000000000003) ".text": DW_OP_reg2 RCX

# BOTH-NEXT:         <default>: DW_OP_reg3 RBX

# REGULAR-NEXT:      [0x0000000000000004, 0x0000000000000005): DW_OP_reg4 RSI
# VERBOSE-NEXT:      [0x0000000000000004, 0x0000000000000005) ".text": DW_OP_reg4 RSI

# REGULAR-NEXT:      [0x0000000000000005, 0x0000000000000006): DW_OP_reg5 RDI
# VERBOSE-NEXT:      [0x0000000000000005, 0x0000000000000006) ".text": DW_OP_reg5 RDI

# REGULAR-NEXT:      [0x0000000000000006, 0x0000000000000007): DW_OP_reg6 RBP
# VERBOSE-NEXT:      [0x0000000000000006, 0x0000000000000007) ".text": DW_OP_reg6 RBP

# REGULAR-NEXT:      [0x0000000000000007, 0x0000000000000008): DW_OP_reg7 RSP
# VERBOSE-NEXT:      [0x0000000000000007, 0x0000000000000008) ".text": DW_OP_reg7 RSP

# BOTH-NEXT:         DW_LLE_startx_length (0x000000000000dead, 0x0000000000000001): DW_OP_reg4 RSI)

# BOTH: locations list header: length = 0x00000056, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000
# BOTH-NEXT: 0x0000000c:
# BOTH-NEXT:     DW_LLE_startx_endx     (0x0000000000000000, 0x0000000000000001): DW_OP_reg0 RAX
# BOTH-NEXT:     DW_LLE_startx_length   (0x0000000000000001, 0x0000000000000001): DW_OP_reg1 RDX
# BOTH-NEXT:     DW_LLE_offset_pair     (0x0000000000000002, 0x0000000000000003): DW_OP_reg2 RCX

# REGULAR-NEXT:  <default>: DW_OP_reg3 RBX
# VERBOSE-NEXT:  DW_LLE_default_location()
# VERBOSE-NEXT:            => <default>: DW_OP_reg3 RBX

# REGULAR-NEXT:  [0x0000000000000004, 0x0000000000000005): DW_OP_reg4 RSI
# VERBOSE-NEXT:  DW_LLE_start_end       (0x0000000000000004, 0x0000000000000005) ".text"
# VERBOSE-NEXT:            => [0x0000000000000004, 0x0000000000000005) ".text": DW_OP_reg4 RSI

# REGULAR-NEXT:  [0x0000000000000005, 0x0000000000000006): DW_OP_reg5 RDI
# VERBOSE-NEXT:  DW_LLE_start_length    (0x0000000000000005, 0x0000000000000001) ".text"
# VERBOSE-NEXT:            => [0x0000000000000005, 0x0000000000000006) ".text": DW_OP_reg5 RDI

# BOTH-NEXT:     DW_LLE_base_addressx   (0x0000000000000002)

# BOTH-NEXT:     DW_LLE_offset_pair     (0x0000000000000000, 0x0000000000000001): DW_OP_reg6 RBP

# VERBOSE-NEXT:  DW_LLE_base_address    (0x0000000000000007) ".text"

# REGULAR-NEXT:  [0x0000000000000007, 0x0000000000000008): DW_OP_reg7 RSP
# VERBOSE-NEXT:  DW_LLE_offset_pair     (0x0000000000000000, 0x0000000000000001)
# VERBOSE-NEXT:            => [0x0000000000000007, 0x0000000000000008) ".text": DW_OP_reg7 RSP

# BOTH-NEXT:     DW_LLE_startx_length   (0x000000000000dead, 0x0000000000000001): DW_OP_reg4 RSI

# VERBOSE-NEXT:  DW_LLE_end_of_list     ()


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
        nop
.Lf5:
        nop
.Lf6:
        nop
.Lf7:
        nop
.Lf8:
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
        .byte   2                       # DW_LLE_startx_endx
        .uleb128 0                      #   start idx
        .uleb128 1                      #   end idx
        .byte   1                       # Loc expr size
        .byte   80                      # super-register DW_OP_reg0

        .byte   3                       # DW_LLE_startx_length
        .uleb128 1                      #   start idx
        .uleb128 .Lf2-.Lf1              #   length
        .byte   1                       # Loc expr size
        .byte   81                      # super-register DW_OP_reg1

        .byte   4                       # DW_LLE_offset_pair
        .uleb128 .Lf2-.Lf0              #   starting offset
        .uleb128 .Lf3-.Lf0              #   ending offset
        .byte   1                       # Loc expr size
        .byte   82                      # super-register DW_OP_reg2

        .byte   5                       # DW_LLE_default_location
        .byte   1                       # Loc expr size
        .byte   83                      # super-register DW_OP_reg3

        .byte   7                       # DW_LLE_start_end
        .quad   .Lf4                    #   starting offset
        .quad   .Lf5                    #   ending offset
        .byte   1                       # Loc expr size
        .byte   84                      # super-register DW_OP_reg4

        .byte   8                       # DW_LLE_start_length
        .quad   .Lf5                    #   starting offset
        .uleb128 .Lf6-.Lf5              #   length
        .byte   1                       # Loc expr size
        .byte   85                      # super-register DW_OP_reg5

        .byte   1                       # DW_LLE_base_addressx
        .uleb128 2                      #   base address

        .byte   4                       # DW_LLE_offset_pair
        .uleb128 .Lf6-.Lf6              #   starting offset
        .uleb128 .Lf7-.Lf6              #   ending offset
        .byte   1                       # Loc expr size
        .byte   86                      # super-register DW_OP_reg6

        .byte   6                       # DW_LLE_base_address
        .quad   .Lf7                    #   base address

        .byte   4                       # DW_LLE_offset_pair
        .uleb128 .Lf7-.Lf7              #   starting offset
        .uleb128 .Lf8-.Lf7              #   ending offset
        .byte   1                       # Loc expr size
        .byte   87                      # super-register DW_OP_reg7

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
        .byte   17                      # DW_AT_low_pc
        .byte   27                      # DW_FORM_addrx
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
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
        .byte   0                       # DW_AT_low_pc
        .long   .Lfend-.Lf0             # DW_AT_high_pc
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
        .quad   .Lf1
        .quad   .Lf6
.Ldebug_addr_end0:
