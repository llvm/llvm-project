# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN:   | not llvm-dwarfdump -lookup=0xffffffff - | \
# RUN: FileCheck %s --check-prefix=EMPTY --allow-empty
# EMPTY: {{^$}}

# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN:   | not llvm-dwarfdump -lookup=0xffffffffffffffff - | \
# RUN: FileCheck %s --check-prefix=EMPTY --allow-empty
# EMPTY: {{^$}}

# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN:   | llvm-dwarfdump -lookup=0x4 - | \
# RUN: FileCheck %s -check-prefixes=CHECK,LEX,A

# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN:   | llvm-dwarfdump -lookup=0xb - | \
# RUN: FileCheck %s -check-prefixes=CHECK,LEX,B

# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN:   | llvm-dwarfdump -lookup=0x14 - | \
# RUN: FileCheck %s -check-prefixes=CHECK,C

# CHECK: Compile Unit: length = 0x00000060, version = 0x0004, abbr_offset = 0x0000, addr_size = 0x08 (next unit at 0x00000064)

# CHECK: DW_TAG_compile_unit
# CHECK:   DW_AT_name        ("foo.c")
# CHECK:   DW_AT_stmt_list   (0x00000000)
# CHECK:   DW_AT_low_pc      (0x0000000000000000)
# CHECK:   DW_AT_high_pc     (0x0000000000000016)

# CHECK: DW_TAG_subprogram
# CHECK:     DW_AT_low_pc    (0x0000000000000000)
# CHECK:     DW_AT_high_pc   (0x0000000000000016)
# CHECK:     DW_AT_name      ("foo")

# LEX: DW_TAG_lexical_block
# LEX:       DW_AT_low_pc  (0x0000000000000004)
# LEX:       DW_AT_high_pc (0x0000000000000014)

# A: Line info: file 'foo.c', line 3, column 9, start line 1
# B: Line info: file 'foo.c', line 4, column 6, start line 1
# C: Line info: file 'foo.c', line 6, column 1, start line 1

	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 13
	.globl	_foo                    ## -- Begin function foo
	.p2align	4, 0x90
_foo:                                   ## @foo
Lfunc_begin0:
	.file	1 "foo.c"
	.loc	1 1 0                   ## foo.c:1:0
	.cfi_startproc
## %bb.0:                               ## %entry
	pushq	%rbp
Lcfi0:
	.cfi_def_cfa_offset 16
Lcfi1:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi2:
	.cfi_def_cfa_register %rbp
Ltmp0:
	.loc	1 3 9 prologue_end      ## foo.c:3:9
	movl	$1, -4(%rbp)
	.loc	1 4 6                   ## foo.c:4:6
	movl	-4(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -4(%rbp)
Ltmp1:
	.loc	1 6 1                   ## foo.c:6:1
	popq	%rbp
	retq
Ltmp2:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang version 6.0.0 (trunk 314509) (llvm/trunk 314517)" ## string offset=0
	.asciz	"foo.c"                 ## string offset=55
	.asciz	"/private/tmp"          ## string offset=61
	.asciz	"foo"                   ## string offset=74
	.asciz	"i"                     ## string offset=78
	.asciz	"int"                   ## string offset=80
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
	.byte	63                      ## DW_AT_external
	.byte	25                      ## DW_FORM_flag_present
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	3                       ## Abbreviation Code
	.byte	11                      ## DW_TAG_lexical_block
	.byte	1                       ## DW_CHILDREN_yes
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	4                       ## Abbreviation Code
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
	.byte	5                       ## Abbreviation Code
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
	.byte	0                       ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
	.long	96                      ## Length of Unit
	.short	4                       ## DWARF version number
Lset0 = Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset0
	.byte	8                       ## Address Size (in bytes)
	.byte	1                       ## Abbrev [1] 0xb:0x59 DW_TAG_compile_unit
	.long	0                       ## DW_AT_producer
	.short	12                      ## DW_AT_language
	.long	55                      ## DW_AT_name
Lset1 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset1
	.long	61                      ## DW_AT_comp_dir
	.quad	Lfunc_begin0            ## DW_AT_low_pc
Lset2 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset2
	.byte	2                       ## Abbrev [2] 0x2a:0x32 DW_TAG_subprogram
	.quad	Lfunc_begin0            ## DW_AT_low_pc
Lset3 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset3
	.byte	1                       ## DW_AT_frame_base
	.byte	86
	.long	74                      ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	1                       ## DW_AT_decl_line
                                        ## DW_AT_external
	.byte	3                       ## Abbrev [3] 0x3f:0x1c DW_TAG_lexical_block
	.quad	Ltmp0                   ## DW_AT_low_pc
Lset4 = Ltmp1-Ltmp0                     ## DW_AT_high_pc
	.long	Lset4
	.byte	4                       ## Abbrev [4] 0x4c:0xe DW_TAG_variable
	.byte	2                       ## DW_AT_location
	.byte	145
	.byte	124
	.long	78                      ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	3                       ## DW_AT_decl_line
	.long	92                      ## DW_AT_type
	.byte	0                       ## End Of Children Mark
	.byte	0                       ## End Of Children Mark
	.byte	5                       ## Abbrev [5] 0x5c:0x7 DW_TAG_base_type
	.long	80                      ## DW_AT_name
	.byte	5                       ## DW_AT_encoding
	.byte	4                       ## DW_AT_byte_size
	.byte	0                       ## End Of Children Mark
	.section	__DWARF,__debug_ranges,regular,debug
Ldebug_range:
	.section	__DWARF,__debug_macinfo,regular,debug
Ldebug_macinfo:
Lcu_macro_begin0:
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
	.long	193491849               ## Hash in Bucket 0
	.long	LNames0-Lnames_begin    ## Offset in Bucket 0
LNames0:
	.long	74                      ## foo
	.long	1                       ## Num DIEs
	.long	42
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
	.long	1                       ## Header Bucket Count
	.long	1                       ## Header Hash Count
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
	.long	193495088               ## Hash in Bucket 0
	.long	Ltypes0-Ltypes_begin    ## Offset in Bucket 0
Ltypes0:
	.long	80                      ## int
	.long	1                       ## Num DIEs
	.long	92
	.short	36
	.byte	0
	.long	0

.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
