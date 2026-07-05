# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj | \
# RUN:   llvm-dwarfdump - | \
# RUN:   FileCheck %s

# This checks that the operand of DW_OP_call_ref is always parsed corresponding
# to the DWARF format of CU. Our code used to have an exception for verson == 2,
# where it treated the operand like it had the size of address, but since
# DW_OP_call_ref was introduced only in DWARF3, the code could be simplified.

# CHECK: DW_TAG_variable
# CHECK-NEXT: DW_AT_location (DW_OP_call_ref 0x11223344)

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	5                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	10                      # DW_FORM_block1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)

	.section	.debug_info,"",@progbits
	.long	.Lcu_end-.Lcu_start     # Length of Unit
.Lcu_start:
	.short	2                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
	.byte	5                       # Abbrev [5] DW_TAG_variable
	.byte	.Lloc_end-.Lloc_begin   # DW_AT_location
.Lloc_begin:
	.byte	154                     # DW_OP_call_ref
	.long	0x11223344              # Offset
.Lloc_end:
	.byte	0                       # End Of Children Mark
.Lcu_end:
