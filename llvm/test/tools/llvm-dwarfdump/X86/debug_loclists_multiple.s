# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump -v %t.o | FileCheck %s

# Test dumping of multiple separate debug_loclist contributions
# CHECK: .debug_loclists contents:
# CHECK: 0x00000000: locations list header:
# CHECK: 0x0000000c:
# CHECK:             DW_LLE_offset_pair (0x0000000000000001, 0x0000000000000002): DW_OP_consts +7, DW_OP_stack_value
# CHECK: 0x00000014: locations list header:
# CHECK:             DW_LLE_offset_pair (0x0000000000000005, 0x0000000000000007): DW_OP_consts +12, DW_OP_stack_value

	.section	.debug_loclists,"",@progbits
	.long	.Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0 # Length
.Ldebug_loclist_table_start0:
	.short	5                       # Version
	.byte	8                       # Address size
	.byte	0                       # Segment selector size
	.long	0                       # Offset entry count

	.byte	4                       # DW_LLE_offset_pair
	.uleb128	1               #   starting offset
	.uleb128	2               #   ending offset
	.byte	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	7                       # 7
	.byte	159                     # DW_OP_stack_value
	.byte	0                       # DW_LLE_end_of_list
.Ldebug_loclist_table_end0:
	.long	.Ldebug_loclist_table_end1-.Ldebug_loclist_table_start1 # Length
.Ldebug_loclist_table_start1:
	.short	5                       # Version
	.byte	8                       # Address size
	.byte	0                       # Segment selector size
	.long	0                       # Offset entry count

	.byte	4                       # DW_LLE_offset_pair
	.uleb128	5               #   starting offset
	.uleb128	7               #   ending offset
	.byte	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	12                      # 12
	.byte	159                     # DW_OP_stack_value
	.byte	0                       # DW_LLE_end_of_list
.Ldebug_loclist_table_end1:
