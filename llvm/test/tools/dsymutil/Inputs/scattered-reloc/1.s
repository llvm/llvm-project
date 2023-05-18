# This assembly file was generated from the following trivial C code:
# $ cat scattered.c 
# int bar = 42;
# $ clang -S -arch armv7 -g scattered.c
# $ clang -c -o 1.o scattered.s
#
# Then I edited the debug info bellow to change the DW_AT_location of the bar
# variable from '.long _bar' to '.long _bar + 16' in order to generate a
# scattered reloc (I do not think LLVM will generate scattered relocs in
# debug info by itself).

	.section	__TEXT,__text,regular,pure_instructions
	.ios_version_min 5, 0
	.syntax unified
	.file	1 "scattered.c"
	.section	__DATA,__data
	.globl	_bar                    @ @bar
	.p2align	2
_bar:
	.long	42                      @ 0x2a

	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang version 3.9.0 (trunk 259311)" @ string offset=0
	.asciz	"scattered.c"           @ string offset=35
	.asciz	"/tmp"                  @ string offset=47
	.asciz	"bar"                   @ string offset=52
	.asciz	"int"                   @ string offset=56
	.section	__DWARF,__debug_loc,regular,debug
Lsection_debug_loc:
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                       @ Abbreviation Code
	.byte	17                      @ DW_TAG_compile_unit
	.byte	1                       @ DW_CHILDREN_yes
	.byte	37                      @ DW_AT_producer
	.byte	14                      @ DW_FORM_strp
	.byte	19                      @ DW_AT_language
	.byte	5                       @ DW_FORM_data2
	.byte	3                       @ DW_AT_name
	.byte	14                      @ DW_FORM_strp
	.byte	16                      @ DW_AT_stmt_list
	.byte	6                       @ DW_FORM_data4
	.byte	27                      @ DW_AT_comp_dir
	.byte	14                      @ DW_FORM_strp
	.byte	0                       @ EOM(1)
	.byte	0                       @ EOM(2)
	.byte	2                       @ Abbreviation Code
	.byte	52                      @ DW_TAG_variable
	.byte	0                       @ DW_CHILDREN_no
	.byte	3                       @ DW_AT_name
	.byte	14                      @ DW_FORM_strp
	.byte	73                      @ DW_AT_type
	.byte	19                      @ DW_FORM_ref4
	.byte	63                      @ DW_AT_external
	.byte	12                      @ DW_FORM_flag
	.byte	58                      @ DW_AT_decl_file
	.byte	11                      @ DW_FORM_data1
	.byte	59                      @ DW_AT_decl_line
	.byte	11                      @ DW_FORM_data1
	.byte	2                       @ DW_AT_location
	.byte	10                      @ DW_FORM_block1
	.byte	0                       @ EOM(1)
	.byte	0                       @ EOM(2)
	.byte	3                       @ Abbreviation Code
	.byte	36                      @ DW_TAG_base_type
	.byte	0                       @ DW_CHILDREN_no
	.byte	3                       @ DW_AT_name
	.byte	14                      @ DW_FORM_strp
	.byte	62                      @ DW_AT_encoding
	.byte	11                      @ DW_FORM_data1
	.byte	11                      @ DW_AT_byte_size
	.byte	11                      @ DW_FORM_data1
	.byte	0                       @ EOM(1)
	.byte	0                       @ EOM(2)
	.byte	0                       @ EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
	.long	52                      @ Length of Unit
	.short	2                       @ DWARF version number
Lset0 = Lsection_abbrev-Lsection_abbrev @ Offset Into Abbrev. Section
	.long	Lset0
	.byte	4                       @ Address Size (in bytes)
	.byte	1                       @ Abbrev [1] 0xb:0x2d DW_TAG_compile_unit
	.long	0                       @ DW_AT_producer
	.short	12                      @ DW_AT_language
	.long	35                      @ DW_AT_name
Lset1 = Lline_table_start0-Lsection_line @ DW_AT_stmt_list
	.long	Lset1
	.long	47                      @ DW_AT_comp_dir
	.byte	2                       @ Abbrev [2] 0x1e:0x12 DW_TAG_variable
	.long	52                      @ DW_AT_name
	.long	48                      @ DW_AT_type
	.byte	1                       @ DW_AT_external
	.byte	1                       @ DW_AT_decl_file
	.byte	1                       @ DW_AT_decl_line
	.byte	5                       @ DW_AT_location
	.byte	3
	.long	_bar + 16
	.byte	3                       @ Abbrev [3] 0x30:0x7 DW_TAG_base_type
	.long	56                      @ DW_AT_name
	.byte	5                       @ DW_AT_encoding
	.byte	4                       @ DW_AT_byte_size
	.byte	0                       @ End Of Children Mark
	.section	__DWARF,__debug_ranges,regular,debug
Ldebug_range:
	.section	__DWARF,__debug_macinfo,regular,debug
	.byte	0                       @ End Of Macro List Mark
	.section	__DWARF,__apple_names,regular,debug
Lnames_begin:
	.long	1212240712              @ Header Magic
	.short	1                       @ Header Version
	.short	0                       @ Header Hash Function
	.long	1                       @ Header Bucket Count
	.long	1                       @ Header Hash Count
	.long	12                      @ Header Data Length
	.long	0                       @ HeaderData Die Offset Base
	.long	1                       @ HeaderData Atom Count
	.short	1                       @ DW_ATOM_die_offset
	.short	6                       @ DW_FORM_data4
	.long	0                       @ Bucket 0
	.long	193487034               @ Hash in Bucket 0
	.long	LNames0-Lnames_begin    @ Offset in Bucket 0
LNames0:
	.long	52                      @ bar
	.long	1                       @ Num DIEs
	.long	30
	.long	0
	.section	__DWARF,__apple_objc,regular,debug
Lobjc_begin:
	.long	1212240712              @ Header Magic
	.short	1                       @ Header Version
	.short	0                       @ Header Hash Function
	.long	1                       @ Header Bucket Count
	.long	0                       @ Header Hash Count
	.long	12                      @ Header Data Length
	.long	0                       @ HeaderData Die Offset Base
	.long	1                       @ HeaderData Atom Count
	.short	1                       @ DW_ATOM_die_offset
	.short	6                       @ DW_FORM_data4
	.long	-1                      @ Bucket 0
	.section	__DWARF,__apple_namespac,regular,debug
Lnamespac_begin:
	.long	1212240712              @ Header Magic
	.short	1                       @ Header Version
	.short	0                       @ Header Hash Function
	.long	1                       @ Header Bucket Count
	.long	0                       @ Header Hash Count
	.long	12                      @ Header Data Length
	.long	0                       @ HeaderData Die Offset Base
	.long	1                       @ HeaderData Atom Count
	.short	1                       @ DW_ATOM_die_offset
	.short	6                       @ DW_FORM_data4
	.long	-1                      @ Bucket 0
	.section	__DWARF,__apple_types,regular,debug
Ltypes_begin:
	.long	1212240712              @ Header Magic
	.short	1                       @ Header Version
	.short	0                       @ Header Hash Function
	.long	1                       @ Header Bucket Count
	.long	1                       @ Header Hash Count
	.long	20                      @ Header Data Length
	.long	0                       @ HeaderData Die Offset Base
	.long	3                       @ HeaderData Atom Count
	.short	1                       @ DW_ATOM_die_offset
	.short	6                       @ DW_FORM_data4
	.short	3                       @ DW_ATOM_die_tag
	.short	5                       @ DW_FORM_data2
	.short	4                       @ DW_ATOM_type_flags
	.short	11                      @ DW_FORM_data1
	.long	0                       @ Bucket 0
	.long	193495088               @ Hash in Bucket 0
	.long	Ltypes0-Ltypes_begin    @ Offset in Bucket 0
Ltypes0:
	.long	56                      @ int
	.long	1                       @ Num DIEs
	.long	48
	.short	36
	.byte	0
	.long	0

.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
