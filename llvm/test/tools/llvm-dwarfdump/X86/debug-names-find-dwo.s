# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj | \
# RUN:   llvm-dwarfdump -find=foobar - | FileCheck %s

# CHECK: DW_TAG_variable
# CHECK-NEXT: DW_AT_name ("foobar")

	.text
	.file	"<stdin>"
	.file	1 "/tmp/cu1.c"
	.type	foobar,@object          # @foobar
	.comm	foobar,8,8
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"foo.dwo"               # string offset=0
.Lskel_string1:
	.asciz	"/tmp"                  # string offset=8
.Lskel_string2:
	.asciz	"foobar"                # string offset=13
	.section	.debug_loc.dwo,"",@progbits
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.ascii	"\260B"                 # DW_AT_GNU_dwo_name
	.byte	14                      # DW_FORM_strp
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.ascii	"\261B"                 # DW_AT_GNU_dwo_id
	.byte	7                       # DW_FORM_data8
	.ascii	"\263B"                 # DW_AT_GNU_addr_base
	.byte	23                      # DW_FORM_sec_offset
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	32                      # Length of Unit
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x19 DW_TAG_compile_unit
	.long	0                       # DW_AT_stmt_list
	.long	.Lskel_string0          # DW_AT_GNU_dwo_name
	.long	.Lskel_string1          # DW_AT_comp_dir
	.quad	-1328675031687321003    # DW_AT_GNU_dwo_id
	.long	.debug_addr             # DW_AT_GNU_addr_base
	.section	.debug_ranges,"",@progbits
	.section	.debug_macinfo,"",@progbits
	.byte	0                       # End Of Macro List Mark
	.section	.debug_str.dwo,"MS",@progbits,1
.Linfo_string0:
	.asciz	"foo.dwo"               # string offset=0
.Linfo_string1:
	.asciz	"clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)" # string offset=8
.Linfo_string2:
	.asciz	"/tmp/cu1.c"            # string offset=63
.Linfo_string3:
	.asciz	"foobar"                # string offset=74
	.section	.debug_str_offsets.dwo,"",@progbits
	.long	0
	.long	8
	.long	63
	.long	74
	.section	.debug_info.dwo,"",@progbits
	.long	34                      # Length of Unit
	.short	4                       # DWARF version number
	.long	0                       # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x1b DW_TAG_compile_unit
	.byte	0                       # DW_AT_GNU_dwo_name
	.byte	1                       # DW_AT_producer
	.short	12                      # DW_AT_language
	.byte	2                       # DW_AT_name
	.quad	-1328675031687321003    # DW_AT_GNU_dwo_id
	.byte	2                       # Abbrev [2] 0x19:0xb DW_TAG_variable
	.byte	3                       # DW_AT_name
	.long	36                      # DW_AT_type
                                        # DW_AT_external
	.byte	1                       # DW_AT_decl_file
	.byte	1                       # DW_AT_decl_line
	.byte	2                       # DW_AT_location
	.byte	251
	.byte	0
	.byte	3                       # Abbrev [3] 0x24:0x1 DW_TAG_pointer_type
	.byte	0                       # End Of Children Mark
	.section	.debug_abbrev.dwo,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.ascii	"\260B"                 # DW_AT_GNU_dwo_name
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.byte	37                      # DW_AT_producer
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.ascii	"\261B"                 # DW_AT_GNU_dwo_id
	.byte	7                       # DW_FORM_data8
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	2                       # DW_AT_location
	.byte	24                      # DW_FORM_exprloc
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	15                      # DW_TAG_pointer_type
	.byte	0                       # DW_CHILDREN_no
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_addr,"",@progbits
	.quad	foobar
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0 # Header: unit length
.Lnames_start0:
	.short	5                       # Header: version
	.short	0                       # Header: padding
	.long	1                       # Header: compilation unit count
	.long	0                       # Header: local type unit count
	.long	0                       # Header: foreign type unit count
	.long	1                       # Header: bucket count
	.long	1                       # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                       # Header: augmentation string size
	.ascii	"LLVM0700"              # Header: augmentation string
	.long	.Lcu_begin0             # Compilation unit 0
	.long	1                       # Bucket 0
	.long	-35364674               # Hash in Bucket 0
	.long	.Lskel_string2          # String in Bucket 0: foobar
	.long	.Lnames0-.Lnames_entries0 # Offset in Bucket 0
.Lnames_abbrev_start0:
	.byte	52                      # Abbrev code
	.byte	52                      # DW_TAG_variable
	.byte	3                       # DW_IDX_die_offset
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames0:
	.byte	52                      # Abbreviation code
	.long	25                      # DW_IDX_die_offset
	.long	0                       # End of list: foobar
	.p2align	2
.Lnames_end0:
