# RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o %t
# RUN: llvm-dwarfdump --debug-info=0x0000002a -p %t | FileCheck %s --check-prefix=PARENTS
# RUN: llvm-dwarfdump --debug-info=0x0000002a -c %t | FileCheck %s --check-prefix=CHILDREN
# RUN: llvm-dwarfdump --debug-info=0x0000002a -p -c %t | FileCheck %s --check-prefix=BOTH
# RUN: llvm-dwarfdump  --debug-info=0x00000032 -p -parent-recurse-depth 1 -c %t | FileCheck %s --check-prefix=ONEPARENT

# PARENTS: DW_TAG_compile_unit
# PARENTS: DW_TAG_enumeration_type
# PARENTS-NOT: DW_TAG_enumerator

# ONEPARENT-NOT: DW_TAG_compile_unit
# ONEPARENT: DW_TAG_enumeration_type
# ONEPARENT: DW_TAG_enumerator

# CHILDREN-NOT: DW_TAG_compile_unit
# CHILDREN:   DW_TAG_enumerator
# CHILDREN:     DW_AT_name	("first")
# CHILDREN:   DW_TAG_enumerator
# CHILDREN:     DW_AT_name	("second")

# BOTH: DW_TAG_compile_unit
# BOTH: DW_TAG_enumeration_type
# BOTH:   DW_TAG_enumerator
# BOTH:     DW_AT_name	("first")
# BOTH:   DW_TAG_enumerator
# BOTH:     DW_AT_name	("second")

# BOTH-NOT: DW_TAG_compile_unit
# BOTH-NOT: DW_TAG_enumeration_type
# BOTH-NOT: DW_TAG_enumerator

	.section	__TEXT,__text,regular,pure_instructions
	.file	1 "/private/tmp" "/tmp/enum.c"
	.globl	_main                   ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
Lfunc_begin0:
	.loc	1 4 0                   ## /tmp/enum.c:4:0
	.cfi_startproc
## %bb.0:                               ## %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, -4(%rbp)
Ltmp0:
	.loc	1 5 7 prologue_end      ## /tmp/enum.c:5:7
	movl	$0, -8(%rbp)
	.loc	1 6 10                  ## /tmp/enum.c:6:10
	movl	-8(%rbp), %eax
	.loc	1 6 3 is_stmt 0         ## /tmp/enum.c:6:3
	popq	%rbp
	retq
Ltmp1:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang version 7.0.0 "  ## string offset=0
	.asciz	"/tmp/enum.c"           ## string offset=21
	.asciz	"/private/tmp"          ## string offset=33
	.asciz	"unsigned int"          ## string offset=46
	.asciz	"first"                 ## string offset=59
	.asciz	"second"                ## string offset=65
	.asciz	"main"                  ## string offset=72
	.asciz	"int"                   ## string offset=77
	.asciz	"var"                   ## string offset=81
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                       ## Abbreviation Code
	.byte	17                      ## DW_TAG_compile_unit
	.byte	1                       ## DW_CHILDREN_yes
	.byte	37                      ## DW_AT_producer
	.byte	14                      ## DW_FORM_strp
	.byte	19                      ## DW_AT_language
	.byte	5                       ## DW_FORM_data2
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	16                      ## DW_AT_stmt_list
	.byte	23                      ## DW_FORM_sec_offset
	.byte	27                      ## DW_AT_comp_dir
	.byte	14                      ## DW_FORM_strp
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	2                       ## Abbreviation Code
	.byte	4                       ## DW_TAG_enumeration_type
	.byte	1                       ## DW_CHILDREN_yes
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	3                       ## Abbreviation Code
	.byte	40                      ## DW_TAG_enumerator
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	28                      ## DW_AT_const_value
	.byte	15                      ## DW_FORM_udata
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	4                       ## Abbreviation Code
	.byte	36                      ## DW_TAG_base_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	62                      ## DW_AT_encoding
	.byte	11                      ## DW_FORM_data1
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	5                       ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	1                       ## DW_CHILDREN_yes
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	64                      ## DW_AT_frame_base
	.byte	24                      ## DW_FORM_exprloc
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	63                      ## DW_AT_external
	.byte	25                      ## DW_FORM_flag_present
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	6                       ## Abbreviation Code
	.byte	52                      ## DW_TAG_variable
	.byte	0                       ## DW_CHILDREN_no
	.byte	2                       ## DW_AT_location
	.byte	24                      ## DW_FORM_exprloc
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	0                       ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
	.long	114                     ## Length of Unit
	.short	4                       ## DWARF version number
.set Lset0, Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset0
	.byte	8                       ## Address Size (in bytes)
	.byte	1                       ## Abbrev [1] 0xb:0x6b DW_TAG_compile_unit
	.long	0                       ## DW_AT_producer
	.short	12                      ## DW_AT_language
	.long	21                      ## DW_AT_name
.set Lset1, Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset1
	.long	33                      ## DW_AT_comp_dir
	.quad	Lfunc_begin0            ## DW_AT_low_pc
.set Lset2, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
	.long	Lset2
	.byte	2                       ## Abbrev [2] 0x2a:0x15 DW_TAG_enumeration_type
	.long	63                      ## DW_AT_type
	.byte	4                       ## DW_AT_byte_size
	.byte	1                       ## DW_AT_decl_file
	.byte	1                       ## DW_AT_decl_line
	.byte	3                       ## Abbrev [3] 0x32:0x6 DW_TAG_enumerator
	.long	59                      ## DW_AT_name
	.byte	0                       ## DW_AT_const_value
	.byte	3                       ## Abbrev [3] 0x38:0x6 DW_TAG_enumerator
	.long	65                      ## DW_AT_name
	.byte	1                       ## DW_AT_const_value
	.byte	0                       ## End Of Children Mark
	.byte	4                       ## Abbrev [4] 0x3f:0x7 DW_TAG_base_type
	.long	46                      ## DW_AT_name
	.byte	7                       ## DW_AT_encoding
	.byte	4                       ## DW_AT_byte_size
	.byte	5                       ## Abbrev [5] 0x46:0x28 DW_TAG_subprogram
	.quad	Lfunc_begin0            ## DW_AT_low_pc
.set Lset3, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
	.long	Lset3
	.byte	1                       ## DW_AT_frame_base
	.byte	86
	.long	72                      ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	3                       ## DW_AT_decl_line
	.long	110                     ## DW_AT_type
                                        ## DW_AT_external
	.byte	6                       ## Abbrev [6] 0x5f:0xe DW_TAG_variable
	.byte	2                       ## DW_AT_location
	.byte	145
	.byte	120
	.long	81                      ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	5                       ## DW_AT_decl_line
	.long	110                     ## DW_AT_type
	.byte	0                       ## End Of Children Mark
	.byte	4                       ## Abbrev [4] 0x6e:0x7 DW_TAG_base_type
	.long	77                      ## DW_AT_name
	.byte	5                       ## DW_AT_encoding
	.byte	4                       ## DW_AT_byte_size
	.byte	0                       ## End Of Children Mark
	.section	__DWARF,__debug_ranges,regular,debug
Ldebug_range:
	.section	__DWARF,__debug_macinfo,regular,debug
Ldebug_macinfo:
	.byte	0                       ## End Of Macro List Mark
	.section	__DWARF,__apple_names,regular,debug
Lnames_begin:
	.long	1212240712              ## Header Magic
	.short	1                       ## Header Version
	.short	0                       ## Header Hash Function
	.long	1                       ## Header Bucket Count
	.long	1                       ## Header Hash Count
	.long	12                      ## Header Data Length
	.long	0                       ## HeaderData Die Offset Base
	.long	1                       ## HeaderData Atom Count
	.short	1                       ## DW_ATOM_die_offset
	.short	6                       ## DW_FORM_data4
	.long	0                       ## Bucket 0
	.long	2090499946              ## Hash in Bucket 0
.set Lset4, LNames0-Lnames_begin        ## Offset in Bucket 0
	.long	Lset4
LNames0:
	.long	72                      ## main
	.long	1                       ## Num DIEs
	.long	70
	.long	0
	.section	__DWARF,__apple_objc,regular,debug
Lobjc_begin:
	.long	1212240712              ## Header Magic
	.short	1                       ## Header Version
	.short	0                       ## Header Hash Function
	.long	1                       ## Header Bucket Count
	.long	0                       ## Header Hash Count
	.long	12                      ## Header Data Length
	.long	0                       ## HeaderData Die Offset Base
	.long	1                       ## HeaderData Atom Count
	.short	1                       ## DW_ATOM_die_offset
	.short	6                       ## DW_FORM_data4
	.long	-1                      ## Bucket 0
	.section	__DWARF,__apple_namespac,regular,debug
Lnamespac_begin:
	.long	1212240712              ## Header Magic
	.short	1                       ## Header Version
	.short	0                       ## Header Hash Function
	.long	1                       ## Header Bucket Count
	.long	0                       ## Header Hash Count
	.long	12                      ## Header Data Length
	.long	0                       ## HeaderData Die Offset Base
	.long	1                       ## HeaderData Atom Count
	.short	1                       ## DW_ATOM_die_offset
	.short	6                       ## DW_FORM_data4
	.long	-1                      ## Bucket 0
	.section	__DWARF,__apple_types,regular,debug
Ltypes_begin:
	.long	1212240712              ## Header Magic
	.short	1                       ## Header Version
	.short	0                       ## Header Hash Function
	.long	2                       ## Header Bucket Count
	.long	2                       ## Header Hash Count
	.long	20                      ## Header Data Length
	.long	0                       ## HeaderData Die Offset Base
	.long	3                       ## HeaderData Atom Count
	.short	1                       ## DW_ATOM_die_offset
	.short	6                       ## DW_FORM_data4
	.short	3                       ## DW_ATOM_die_tag
	.short	5                       ## DW_FORM_data2
	.short	4                       ## DW_ATOM_type_flags
	.short	11                      ## DW_FORM_data1
	.long	0                       ## Bucket 0
	.long	1                       ## Bucket 1
	.long	193495088               ## Hash in Bucket 0
	.long	-1304652851             ## Hash in Bucket 1
.set Lset5, Ltypes1-Ltypes_begin        ## Offset in Bucket 0
	.long	Lset5
.set Lset6, Ltypes0-Ltypes_begin        ## Offset in Bucket 1
	.long	Lset6
Ltypes1:
	.long	77                      ## int
	.long	1                       ## Num DIEs
	.long	110
	.short	36
	.byte	0
	.long	0
Ltypes0:
	.long	46                      ## unsigned int
	.long	1                       ## Num DIEs
	.long	63
	.short	36
	.byte	0
	.long	0

.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
