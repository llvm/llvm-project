# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o %t.o
# RUN: not llvm-dwarfdump -v -debug-info -debug-line -debug-addr -debug-rnglists -debug-ranges %t.o | FileCheck --implicit-check-not=DW_TAG --implicit-check-not=DW_AT %s

# FIXME: Remove the 'not' once the rnglist are lazily/correctly parsed (see comment below)

# Test that llvm - dwarfdump strips addresses relating to dead code(using the
# DWARFv6 - proposed tombstone constant & nearest equivalent for debug_ranges)
# Testing the tombstone use in debug_info (addr/addrx), debug_ranges,
# debug_rnglists, debug_ranges, and debug_line.

# CHECK-DAG: .debug_info contents:
# CHECK:     DW_TAG_compile_unit
# CHECK:       DW_AT_ranges [DW_FORM_sec_offset] (0x00000000
# CHECK-NEXT:    [0x00000042, 0x00000048))
# CHECK:       DW_TAG_subprogram
# CHECK:         DW_AT_low_pc [DW_FORM_addr]     (0xffffffff (dead code))
# CHECK:         DW_AT_high_pc [DW_FORM_data4]   (0x00000006)
# CHECK:       DW_TAG_subprogram
# CHECK:         DW_AT_low_pc [DW_FORM_addr]     (0x00000042)
# CHECK:         DW_AT_high_pc [DW_FORM_data4]   (0x00000006)
# CHECK:     DW_TAG_compile_unit
# CHECK:       DW_AT_addr_base
# CHECK:       DW_AT_ranges [DW_FORM_sec_offset] (0x0000000c
# CHECK-NEXT:    [0x00000042, 0x00000048)
# CHECK-NEXT:    [0x00000042, 0x00000048)
# CHECK-NEXT:    [0x00000042, 0x00000048)
# CHECK-NEXT:    [0x00000042, 0x00000048)
# CHECK-NEXT:    [0x00000042, 0x00000048))
# CHECK:       DW_TAG_subprogram
# CHECK:         DW_AT_low_pc [DW_FORM_addrx]     (indexed (00000000) address = 0xffffffff (dead code))
# CHECK:         DW_AT_high_pc [DW_FORM_data4]   (0x00000006)
# CHECK:       DW_TAG_subprogram
# CHECK:         DW_AT_low_pc [DW_FORM_addrx]     (indexed (00000001) address = 0x00000042)
# CHECK:         DW_AT_high_pc [DW_FORM_data4]   (0x00000006)
# CHECK:     DW_TAG_compile_unit
# CHECK:       DW_AT_ranges [DW_FORM_sec_offset] (0x00000018
# CHECK-NEXT:    [0x0000000000000042, 0x0000000000000048))
# CHECK:       DW_TAG_subprogram
# CHECK:         DW_AT_low_pc [DW_FORM_addr]     (0xffffffffffffffff (dead code))
# CHECK:         DW_AT_high_pc [DW_FORM_data4]   (0x00000006)
# CHECK:       DW_TAG_subprogram
# CHECK:         DW_AT_low_pc [DW_FORM_addr]     (0x0000000000000042)
# CHECK:         DW_AT_high_pc [DW_FORM_data4]   (0x00000006)
# CHECK:     DW_TAG_compile_unit
# CHECK:       DW_AT_addr_base

# FIXME: Lazily parse rnglists rather than expecting to be able to parse an
#        entire rnglists contribution (since there's no way to know where such a
#        contribution starts) - rather than assuming one starts at 0.

# CHECK:       DW_AT_ranges [DW_FORM_sec_offset] (0x00000057)
#     [0x0000000000000042, 0x0000000000000048)
#     [0x0000000000000042, 0x0000000000000048)
#     [0x0000000000000042, 0x0000000000000048)
#     [0x0000000000000042, 0x0000000000000048)
#     [0x0000000000000042, 0x0000000000000048))
# CHECK:       DW_TAG_subprogram
# CHECK:         DW_AT_low_pc [DW_FORM_addrx]     (indexed (00000000) address = 0xffffffffffffffff (dead code))
# CHECK:         DW_AT_high_pc [DW_FORM_data4]   (0x00000006)
# CHECK:       DW_TAG_subprogram
# CHECK:         DW_AT_low_pc [DW_FORM_addrx]     (indexed (00000001) address = 0x0000000000000042)
# CHECK:         DW_AT_high_pc [DW_FORM_data4]   (0x00000006)

# CHECK-DAG: .debug_line contents:
# CHECK:      Address Line
# CHECK-NEXT: --------------
# CHECK-NEXT: DW_LNE_set_address (0xffffffff)
# CHECK-NEXT: DW_LNS_copy
# CHECK-NEXT: DW_LNS_advance_pc (1)
# CHECK-NEXT: DW_LNE_end_sequence
# CHECK-NEXT: DW_LNE_set_address (0x00000042)
# CHECK-NEXT: DW_LNS_copy
# CHECK-NEXT:   0x0000000000000042 1
# CHECK-NEXT: DW_LNS_advance_pc (1)
# CHECK-NEXT: DW_LNE_end_sequence
# CHECK:      Address Line
# CHECK-NEXT: --------------
# CHECK-NEXT: DW_LNE_set_address (0xffffffffffffffff)
# CHECK-NEXT: DW_LNS_copy
# CHECK-NEXT: DW_LNS_advance_pc (1)
# CHECK-NEXT: DW_LNE_end_sequence
# CHECK-NEXT: DW_LNE_set_address (0x0000000000000042)
# CHECK-NEXT: DW_LNS_copy
# CHECK-NEXT:   0x0000000000000042 1
# CHECK-NEXT: DW_LNS_advance_pc (1)
# CHECK-NEXT: DW_LNE_end_sequence

# Dumping of the debug_addr, ranges, and rnglists sections don't do anything
# different with tombstoned addresses, but dump them just for
# documentation/comparison with the tombstone-filtered renderings in the
# debug_info section above

# CHECK-DAG: .debug_addr contents:
# CHECK-NEXT: addr_size = 0x04
# CHECK-NEXT: Addrs: [
# CHECK-NEXT: 0xffffffff
# CHECK-NEXT: 0x00000042
# CHECK-NEXT: ]
# CHECK-NEXT: addr_size = 0x08
# CHECK-NEXT: Addrs: [
# CHECK-NEXT: 0xffffffffffffffff
# CHECK-NEXT: 0x0000000000000042
# CHECK-NEXT: ]

# CHECK-DAG: .debug_ranges contents:
# CHECK-NEXT: fffffffe fffffffe
# CHECK-NEXT: 00000042 00000048
# CHECK-NEXT: <End of list>
# FIXME: Would be nice if we didn't assume all the contributions were of the
#        same address size, instead dumping them based on the address size of
#        the unit that references them. Maybe optimistically guessing at any
#        unreferenced chunks. (this would be more like libdwarf/dwarfdump).
#        But for now, these 64bit address ranges are mangled/being rendered
#        here as though they were a 32 bit address range.
# CHECK-NEXT: fffffffe ffffffff
# CHECK-NEXT: fffffffe ffffffff
# CHECK-NEXT: 00000042 00000000
# CHECK-NEXT: 00000048 00000000
# CHECK-NEXT: <End of list>

# CHECK-DAG: .debug_rnglists contents:
# CHECK-NEXT: addr_size = 0x04
# CHECK-NEXT: ranges:
# CHECK-NEXT: [DW_RLE_start_length ]:  0xffffffff, 0x00000006
# CHECK-NEXT: [DW_RLE_start_length ]:  0x00000042, 0x00000006
# CHECK-NEXT: [DW_RLE_startx_length]:  0x00000000, 0x00000006
# CHECK-NEXT: [DW_RLE_startx_length]:  0x00000001, 0x00000006
# CHECK-NEXT: [DW_RLE_start_end    ]: [0xffffffff, 0xffffffff)
# CHECK-NEXT: [DW_RLE_start_end    ]: [0x00000042, 0x00000048)
# CHECK-NEXT: [DW_RLE_base_address ]:  0x00000040
# CHECK-NEXT: [DW_RLE_offset_pair  ]:  0x00000002, 0x00000008 => [0x00000042, 0x00000048)
# CHECK-NEXT: [DW_RLE_base_address ]:  0xffffffff
# CHECK-NEXT: [DW_RLE_offset_pair  ]:  0x00000002, 0x00000008 => dead code
# CHECK-NEXT: [DW_RLE_base_addressx]:  0x00000000
# FIXME: Don't print "computed" values that aren't really computed/instead
#        still refer to the index instead of the resulting address
# CHECK-NEXT: [DW_RLE_offset_pair  ]:  0x00000000, 0x00000006 => [0x00000000, 0x00000006)
# CHECK-NEXT: [DW_RLE_base_addressx]:  0x00000001
# CHECK-NEXT: [DW_RLE_offset_pair  ]:  0x00000000, 0x00000006 => [0x00000001, 0x00000007)
# CHECK-NEXT: [DW_RLE_end_of_list  ]
# CHECK-NEXT: addr_size = 0x08
# CHECK-NEXT: ranges:
# CHECK-NEXT: [DW_RLE_start_length ]:  0xffffffffffffffff, 0x0000000000000006
# CHECK-NEXT: [DW_RLE_start_length ]:  0x0000000000000042, 0x0000000000000006
# CHECK-NEXT: [DW_RLE_startx_length]:  0x0000000000000000, 0x0000000000000006
# CHECK-NEXT: [DW_RLE_startx_length]:  0x0000000000000001, 0x0000000000000006
# CHECK-NEXT: [DW_RLE_start_end    ]: [0xffffffffffffffff, 0xffffffffffffffff)
# CHECK-NEXT: [DW_RLE_start_end    ]: [0x0000000000000042, 0x0000000000000048)
# CHECK-NEXT: [DW_RLE_base_address ]:  0x0000000000000040
# CHECK-NEXT: [DW_RLE_offset_pair  ]:  0x0000000000000002, 0x0000000000000008 => [0x0000000000000042, 0x0000000000000048)
# CHECK-NEXT: [DW_RLE_base_address ]:  0xffffffffffffffff
# CHECK-NEXT: [DW_RLE_offset_pair  ]:  0x0000000000000002, 0x0000000000000008 => dead code
# CHECK-NEXT: [DW_RLE_base_addressx]:  0x0000000000000000
# CHECK-NEXT: [DW_RLE_offset_pair  ]:  0x0000000000000000, 0x0000000000000006 => [0x0000000000000000, 0x0000000000000006)
# CHECK-NEXT: [DW_RLE_base_addressx]:  0x0000000000000001
# CHECK-NEXT: [DW_RLE_offset_pair  ]:  0x0000000000000000, 0x0000000000000006 => [0x0000000000000001, 0x0000000000000007)
# CHECK-NEXT: [DW_RLE_end_of_list  ]

	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev4:
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
.Ldebug_abbrev5:
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)

	.section	.debug_info,"",@progbits
	.long	.Ldebug_info4_end-.Ldebug_info4_begin # Length of Unit
.Ldebug_info4_begin:
	.short	4                               # DWARF version number
	.long	.Ldebug_abbrev4                 # Offset Into Abbrev. Section
	.byte	4                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x4a DW_TAG_compile_unit
	.long	.Ldebug_ranges                  # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x2a:0x15 DW_TAG_subprogram
	.long	0xffffffff                      # DW_AT_low_pc
	.long	0x6                             # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x3f:0x15 DW_TAG_subprogram
	.long	0x42                            # DW_AT_low_pc
	.long	0x6                             # DW_AT_high_pc
	.byte	0                               # End Of Children Mark
.Ldebug_info4_end:
	.long	.Ldebug_info5_end-.Ldebug_info5_begin # Length of Unit
.Ldebug_info5_begin:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	4                               # Address Size (in bytes)
	.long	.Ldebug_abbrev5                 # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xb:0x4a DW_TAG_compile_unit
	.long	.Ldebug_addr_base               # DW_AT_addr_base
	.long	.Ldebug_rnglists                # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x2a:0x15 DW_TAG_subprogram
	.uleb128 0                              # DW_AT_low_pc
	.long	0x6                             # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x3f:0x15 DW_TAG_subprogram
	.uleb128 1                              # DW_AT_low_pc
	.long	0x6                             # DW_AT_high_pc
	.byte	0                               # End Of Children Mark
.Ldebug_info5_end:
	.long	.Ldebug_info4_64_end-.Ldebug_info4_64_begin # Length of Unit
.Ldebug_info4_64_begin:
	.short	4                               # DWARF version number
	.long	.Ldebug_abbrev4                 # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x4a DW_TAG_compile_unit
	.long	.Ldebug_ranges_64               # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x2a:0x15 DW_TAG_subprogram
	.quad	0xffffffffffffffff              # DW_AT_low_pc
	.long	0x6                             # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x3f:0x15 DW_TAG_subprogram
	.quad	0x42                            # DW_AT_low_pc
	.long	0x6                             # DW_AT_high_pc
	.byte	0                               # End Of Children Mark
.Ldebug_info4_64_end:
	.long	.Ldebug_info5_64_end-.Ldebug_info5_64_begin # Length of Unit
.Ldebug_info5_64_begin:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.Ldebug_abbrev5                 # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xb:0x4a DW_TAG_compile_unit
	.long	.Ldebug_addr_64_base            # DW_AT_addr_base
	.long	.Ldebug_rnglists_64             # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x2a:0x15 DW_TAG_subprogram
	.uleb128 0                              # DW_AT_low_pc
	.long	0x6                             # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x3f:0x15 DW_TAG_subprogram
	.uleb128 1                              # DW_AT_low_pc
	.long	0x6                             # DW_AT_high_pc
	.byte	0                               # End Of Children Mark
.Ldebug_info5_64_end:

	.section	.debug_ranges,"",@progbits
.Ldebug_ranges:
	.long	0xfffffffe
	.long	0xfffffffe
	.long	0x42
	.long	0x48
	.long	0
	.long	0
.Ldebug_ranges_64:
	.quad	0xfffffffffffffffe
	.quad	0xfffffffffffffffe
	.quad	0x42
	.quad	0x48
	.quad	0
	.quad	0

	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_rnglists_end-.Ldebug_rnglists_begin # Length
.Ldebug_rnglists_begin:
	.short	5                               # Version
	.byte	4                               # Address size
	.byte	0                               # Segment selector size
	.long	0                               # Offset entry count
.Ldebug_rnglists:
	.byte	7                               # DW_RLE_start_length
	.long	0xffffffff                      #   start address
	.uleb128 0x6                            #   length
	.byte	7                               # DW_RLE_start_length
	.long	0x42                            #   start address
	.uleb128 0x6                            #   length
	.byte	3                               # DW_RLE_startx_length
	.uleb128 0                              #   start index
	.uleb128 0x6                            #   length
	.byte	3                               # DW_RLE_startx_length
	.uleb128 1                              #   start index
	.uleb128 0x6                            #   length
	.byte	6                               # DW_RLE_start_end
	.long	0xffffffff                      #   start address
	.long	0xffffffff                      #   end address
	.byte	6                               # DW_RLE_start_end
	.long	0x42                            #   start address
	.long   0x48                            #   length
# FIXME: RLE_startx_endx unsupported by llvm-dwarfdump
#	.byte	2                               # DW_RLE_startx_endx
#	.uleb128 0                              #   start address
#	.uleb128 0                              #   length
#	.byte	2                               # DW_RLE_startx_endx
#	.uleb128 1                              #   start address
#	.uleb128 1                              #   length
	.byte	5                               # DW_RLE_base_address
	.long	0x40                            #   address
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 2                              #   start offset
	.uleb128 8                              #   end offset
	.byte	5                               # DW_RLE_base_address
	.long	0xffffffff                      #   address
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 2                              #   start offset
	.uleb128 8                              #   end offset
	.byte	1                               # DW_RLE_base_addressx
	.uleb128 0                              #   address
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 0                              #   start offset
	.uleb128 6                              #   end offset
	.byte	1                               # DW_RLE_base_addressx
	.uleb128 1                              #   address
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 0                              #   start offset
	.uleb128 6                              #   end offset
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_rnglists_end:
	.long	.Ldebug_rnglists_64_end-.Ldebug_rnglists_64_begin # Length
.Ldebug_rnglists_64_begin:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	0                               # Offset entry count
.Ldebug_rnglists_64:
	.byte	7                               # DW_RLE_start_length
	.quad	0xffffffffffffffff              #   start address
	.uleb128 0x6                            #   length
	.byte	7                               # DW_RLE_start_length
	.quad	0x42                            #   start address
	.uleb128 0x6                            #   length
	.byte	3                               # DW_RLE_startx_length
	.uleb128 0                              #   start index
	.uleb128 0x6                            #   length
	.byte	3                               # DW_RLE_startx_length
	.uleb128 1                              #   start index
	.uleb128 0x6                            #   length
	.byte	6                               # DW_RLE_start_end
	.quad	0xffffffffffffffff              #   start address
	.quad	0xffffffffffffffff              #   end address
	.byte	6                               # DW_RLE_start_end
	.quad	0x42                            #   start address
	.quad   0x48                            #   length
# FIXME: RLE_startx_endx unsupported by llvm-dwarfdump
#	.byte	2                               # DW_RLE_startx_endx
#	.uleb128 0                              #   start address
#	.uleb128 0                              #   length
#	.byte	2                               # DW_RLE_startx_endx
#	.uleb128 1                              #   start address
#	.uleb128 1                              #   length
	.byte	5                               # DW_RLE_base_address
	.quad	0x40                            #   address
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 2                              #   start offset
	.uleb128 8                              #   end offset
	.byte	5                               # DW_RLE_base_address
	.quad	0xffffffffffffffff              #   address
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 2                              #   start offset
	.uleb128 8                              #   end offset
	.byte	1                               # DW_RLE_base_addressx
	.uleb128 0                              #   address
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 0                              #   start offset
	.uleb128 6                              #   end offset
	.byte	1                               # DW_RLE_base_addressx
	.uleb128 1                              #   address
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 0                              #   start offset
	.uleb128 6                              #   end offset
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_rnglists_64_end:

	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end-.Ldebug_addr_begin # Length of contribution
.Ldebug_addr_begin:
	.short	5                               # DWARF version number
	.byte	4                               # Address size
	.byte	0                               # Segment selector size
.Ldebug_addr_base:
	.long	0xffffffff
	.long	0x42
.Ldebug_addr_end:
	.long	.Ldebug_addr_64_end-.Ldebug_addr_64_begin # Length of contribution
.Ldebug_addr_64_begin:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Ldebug_addr_64_base:
	.quad	0xffffffffffffffff
	.quad	0x42
.Ldebug_addr_64_end:

	.section	.debug_line,"",@progbits
.Ldebug_line5:
	.long   .Ldebug_line5_end-.Ldebug_line5_begin   # Length of Unit (DWARF-32 format)
.Ldebug_line5_begin:
	.short  5               # DWARF version number
	.byte   4               # Address Size
	.byte   0               # Segment Selector Size
	.long   .Ldebug_line5_header_end-.Ldebug_line5_header_begin     # Length of Prologue
.Ldebug_line5_header_begin:
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
	.byte   0x08            # DW_FORM_string
	# Directory table entries
	.byte   1               # Two directory entries
	.asciz "dir1"
	# File table format
	.byte   2               # Four elements per file entry
	.byte   2               # DW_LNCT_directory_index
	.byte   0x0b            # DW_FORM_data1
	.byte   1               # DW_LNCT_path
	.byte   0x08            # DW_FORM_string
	# File table entries
	.byte   1               # Two file entries
	.byte   1
	.asciz   "file1"
.Ldebug_line5_header_end:
	.byte   0               # Extended opcode
	.byte   5               #   Size 5
	.byte   2               #   Opcode: DW_LNE_set_address
	.long   0xffffffff      #     address
	.byte	1               # DW_LNS_copy
	.byte	2               # DW_LNS_advance_pc
	.uleb128 1              #   instruction increment
	.byte   0               # Extended opcode
	.byte   1               #   Size 1
	.byte   1               #   Opcode: DW_LNE_end_sequence
	.byte   0               # Extended opcode
	.byte   5               #   Size 5
	.byte   2               #   Opcode: DW_LNE_set_address
	.long   0x42            #     address
	.byte	1               # DW_LNS_copy
	.byte	2               # DW_LNS_advance_pc
	.uleb128 1              #   instruction increment
	.byte   0               # Extended opcode
	.byte   1               #   Size 1
	.byte   1               #   Opcode: DW_LNE_end_sequence
.Ldebug_line5_end:

.Ldebug_line5_64:
	.long   .Ldebug_line5_64_end-.Ldebug_line5_64_begin   # Length of Unit (DWARF-32 format)
.Ldebug_line5_64_begin:
	.short  5               # DWARF version number
	.byte   8               # Address Size
	.byte   0               # Segment Selector Size
	.long   .Ldebug_line5_64_header_end-.Ldebug_line5_64_header_begin     # Length of Prologue
.Ldebug_line5_64_header_begin:
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
	.byte   0x08            # DW_FORM_string
	# Directory table entries
	.byte   1               # Two directory entries
	.asciz "dir1"
	# File table format
	.byte   2               # Four elements per file entry
	.byte   2               # DW_LNCT_directory_index
	.byte   0x0b            # DW_FORM_data1
	.byte   1               # DW_LNCT_path
	.byte   0x08            # DW_FORM_string
	# File table entries
	.byte   1               # Two file entries
	.byte   1
	.asciz   "file1"
.Ldebug_line5_64_header_end:
	.byte   0               # Extended opcode
	.byte   9               #   Size 9
	.byte   2               #   Opcode: DW_LNE_set_address
	.quad   0xffffffffffffffff #     address
	.byte	1               # DW_LNS_copy
	.byte	2               # DW_LNS_advance_pc
	.uleb128 1              #   instruction increment
	.byte   0               # Extended opcode
	.byte   1               #   Size 1
	.byte   1               #   Opcode: DW_LNE_end_sequence
	.byte   0               # Extended opcode
	.byte   9               #   Size 9
	.byte   2               #   Opcode: DW_LNE_set_address
	.quad   0x42            #     address
	.byte	1               # DW_LNS_copy
	.byte	2               # DW_LNS_advance_pc
	.uleb128 1              #   instruction increment
	.byte   0               # Extended opcode
	.byte   1               #   Size 1
	.byte   1               #   Opcode: DW_LNE_end_sequence
.Ldebug_line5_64_end:

