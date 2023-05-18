# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump --debug-loc -v %t.o | FileCheck %s

# CHECK:         .debug_loc contents:
# CHECK-NEXT:    0x00000000:
# CHECK-NEXT:    (0xffffffffffffffff, 0x000000000000002a)
# CHECK-NEXT:    (0x0000000000000003, 0x0000000000000007)
# CHECK-NEXT:        => [0x000000000000002d, 0x0000000000000031): DW_OP_consts +3, DW_OP_stack_value

	.section	.debug_loc,"",@progbits
	.quad	0xffffffffffffffff
	.quad	42
	.quad	3
	.quad	7
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	3                       # 3
	.byte	159                     # DW_OP_stack_value
	.quad	0
	.quad	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
.Ldebug_info_end0:
