# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj | llvm-dwarfdump - | FileCheck %s
#
# CHECK: DW_TAG_variable
# CHECK-NEXT: DW_AT_name ("a")
# CHECK-NEXT: DW_AT_location
# CHECK-NEXT: DW_OP_GNU_entry_value(DW_OP_reg5 RDI), DW_OP_stack_value)

	.section	.debug_str,"MS",@progbits,1
.Linfo_producer:
	.asciz	"hand-written DWARF"
.Lname_a:
	.asciz	"a"

	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	0
	.quad	1
	.short	.Lloc0_end-.Lloc0_start # Loc expr size
.Lloc0_start:
	.byte	243                     # DW_OP_GNU_entry_value
	.byte	1                       # 1
	.byte	85                      # super-register DW_OP_reg5
	.byte	159                     # DW_OP_stack_value
.Lloc0_end:
	.quad	0
	.quad	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	5                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	2                       # DW_AT_location
	.byte	23                      # DW_FORM_sec_offset
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)

	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Lcu_end0-.Lcu_start0   # Length of Unit
.Lcu_start0:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
	.long	.Linfo_producer         # DW_AT_producer
	.byte	5                       # Abbrev [5] DW_TAG_variable
	.long	.Lname_a                # DW_AT_name
	.long	.Ldebug_loc0            # DW_AT_location
	.byte	0                       # End Of Children Mark
.Lcu_end0:
