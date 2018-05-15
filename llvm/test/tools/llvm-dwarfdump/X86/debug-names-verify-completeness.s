# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj -o - | not llvm-dwarfdump -verify - | FileCheck %s

# CHECK: error: Name Index @ 0x0: Entry for DIE @ 0x10 (DW_TAG_namespace) with name namesp missing.
# CHECK: error: Name Index @ 0x0: Entry for DIE @ 0x15 (DW_TAG_variable) with name var_block_addr missing.
# CHECK: error: Name Index @ 0x0: Entry for DIE @ 0x25 (DW_TAG_namespace) with name (anonymous namespace) missing.
# CHECK: error: Name Index @ 0x0: Entry for DIE @ 0x26 (DW_TAG_variable) with name var_loc_addr missing.
# CHECK: error: Name Index @ 0x0: Entry for DIE @ 0x30 (DW_TAG_variable) with name var_loc_tls missing.
# CHECK: error: Name Index @ 0x0: Entry for DIE @ 0x37 (DW_TAG_variable) with name var_loc_gnu_tls missing.
# CHECK: error: Name Index @ 0x0: Entry for DIE @ 0x3e (DW_TAG_subprogram) with name fun_name missing.
# CHECK: error: Name Index @ 0x0: Entry for DIE @ 0x3e (DW_TAG_subprogram) with name _Z8fun_name missing.
# CHECK: error: Name Index @ 0x0: Entry for DIE @ 0x4f (DW_TAG_inlined_subroutine) with name fun_inline missing.
# CHECK: error: Name Index @ 0x0: Entry for DIE @ 0x64 (DW_TAG_label) with name label missing.

	.section	.debug_str,"MS",@progbits,1
.Linfo_producer:
	.asciz	"hand-written DWARF"
.Lname_var_block_addr:
	.asciz	"var_block_addr"
.Lname_var_loc_addr:
	.asciz	"var_loc_addr"
.Lname_var_loc_tls:
	.asciz	"var_loc_tls"
.Lname_var_loc_gnu_tls:
	.asciz	"var_loc_gnu_tls"
.Lname_fun_name:
	.asciz	"fun_name"
.Lname_fun_link_name:
	.asciz	"_Z8fun_name"
.Lname_fun_inline:
	.asciz	"fun_inline"
.Lnamespace:
	.asciz	"namesp"
.Lname_label:
	.asciz	"label"

	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	0
	.quad	1
	.short	.Lloc0_end-.Lloc0_start # Loc expr size
.Lloc0_start:
	.byte	3                       # DW_OP_addr
        .quad 0x47
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

	.byte	2                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	2                       # DW_AT_location
	.byte	24                      # DW_FORM_exprloc
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)

	.byte	3                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	110                     # DW_AT_linkage_name
	.byte	14                      # DW_FORM_strp
	.byte	82                      # DW_AT_entry_pc
	.byte	1                       # DW_FORM_addr
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)

	.byte	4                       # Abbreviation Code
	.byte	57                      # DW_TAG_namespace
	.byte	1                       # DW_CHILDREN_yes
	.byte	3                       # DW_AT_name
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

	.byte	6                       # Abbreviation Code
	.byte	57                      # DW_TAG_namespace
	.byte	1                       # DW_CHILDREN_yes
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)

	.byte	7                       # Abbreviation Code
	.byte	29                      # DW_TAG_inlined_subroutine
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	1                       # DW_FORM_addr
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)

	.byte	8                       # Abbreviation Code
	.byte	10                      # DW_TAG_label
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	82                      # DW_AT_entry_pc
	.byte	1                       # DW_FORM_addr
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

	.byte	4                       # Abbrev [4] DW_TAG_namespace
	.long	.Lnamespace             # DW_AT_name
	.byte	2                       # Abbrev [2] DW_TAG_variable
	.long	.Lname_var_block_addr   # DW_AT_name
	.byte	9                       # DW_AT_location
	.byte	3                       # DW_OP_addr
	.quad	0x47
	.byte	0                       # End Of Children Mark

	.byte	6                       # Abbrev [6] DW_TAG_namespace
	.byte	5                       # Abbrev [5] DW_TAG_variable
	.long	.Lname_var_loc_addr     # DW_AT_name
	.long	.Ldebug_loc0            # DW_AT_location
	.byte	0                       # End Of Children Mark

	.byte	2                       # Abbrev [2] DW_TAG_variable
	.long	.Lname_var_loc_tls      # DW_AT_name
	.byte	1                       # DW_AT_location
	.byte	0x9b                    # DW_OP_form_tls_address

	.byte	2                       # Abbrev [2] DW_TAG_variable
	.long	.Lname_var_loc_gnu_tls  # DW_AT_name
	.byte	1                       # DW_AT_location
	.byte	0xe0                    # DW_OP_GNU_push_tls_address

	.byte	3                       # Abbrev [3] DW_TAG_subprogram
	.long	.Lname_fun_name         # DW_AT_name
	.long	.Lname_fun_link_name    # DW_AT_linkage_name
	.quad	0x47                    # DW_AT_entry_pc
	.byte	7                       # Abbrev [7] DW_TAG_inlined_subroutine
	.long	.Lname_fun_inline       # DW_AT_name
	.quad	0x48                    # DW_AT_low_pc
	.quad	0x49                    # DW_AT_high_pc
	.byte	8                       # Abbrev [8] DW_TAG_label
	.long	.Lname_label            # DW_AT_name
	.quad	0x4a                    # DW_AT_entry_pc
	.byte	0                       # End Of Children Mark

	.byte	0                       # End Of Children Mark
.Lcu_end0:

	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0 # Header: contribution length
.Lnames_start0:
	.short	5                       # Header: version
	.short	0                       # Header: padding
	.long	1                       # Header: compilation unit count
	.long	0                       # Header: local type unit count
	.long	0                       # Header: foreign type unit count
	.long	0                       # Header: bucket count
	.long	0                       # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	0                       # Header: augmentation length
	.long	.Lcu_begin0             # Compilation unit 0
.Lnames_abbrev_start0:
	.byte	0                       # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames_end0:

