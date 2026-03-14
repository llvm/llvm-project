// REQUIRES: x86-registered-target

// RUN: llvm-mc -filetype=obj -triple=i386-linux-gnu -o %t.o %s
// RUN: echo 'FRAME %t.o 0' | llvm-symbolizer | FileCheck %s

// CHECK: f
// CHECK-NEXT: a
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:11
// CHECK-NEXT: -5 1 ??
// CHECK-NEXT: f
// CHECK-NEXT: b
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:12
// CHECK-NEXT: -12 4 ??
// CHECK-NEXT: f
// CHECK-NEXT: c
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:13
// CHECK-NEXT: -16 4 ??
// CHECK-NEXT: f
// CHECK-NEXT: d
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:14
// CHECK-NEXT: -20 4 ??
// CHECK-NEXT: f
// CHECK-NEXT: e
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:15
// CHECK-NEXT: -32 8 ??
// CHECK-NEXT: f
// CHECK-NEXT: f
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:16
// CHECK-NEXT: -36 4 ??
// CHECK-NEXT: f
// CHECK-NEXT: g
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:17
// CHECK-NEXT: -37 1 ??
// CHECK-NEXT: f
// CHECK-NEXT: h
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:18
// CHECK-NEXT: -38 1 ??
// CHECK-NEXT: f
// CHECK-NEXT: i
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:19
// CHECK-NEXT: -44 4 ??
// CHECK-NEXT: f
// CHECK-NEXT: j
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:21
// CHECK-NEXT: -45 1 ??
// CHECK-NEXT: f
// CHECK-NEXT: k
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:22
// CHECK-NEXT: -57 12 ??
// CHECK-NEXT: f
// CHECK-NEXT: l
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:23
// CHECK-NEXT: -345 288 ??
// CHECK-NEXT: f
// CHECK-NEXT: m
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:24
// CHECK-NEXT: -352 4 ??

// Generated from:
//
// struct S;
//
// void f() {
//   char a;
//   char *b;
//   char &c = a;
//   char &&d = 1;
//   char (S::*e)();
//   char S::*f;
//   const char g = 2;
//   volatile char h;
//   char *__restrict i;
//   typedef char char_typedef;
//   char_typedef j;
//   char k[12];
//   char l[12][24];
// }
//
// clang++ --target=i386-linux-gnu frame-types.cpp -g -std=c++11 -S -o frame-types.s 

	.text
	.file	"frame-types.cpp"
	.globl	_Z1fv                           # -- Begin function _Z1fv
	.p2align	4, 0x90
	.type	_Z1fv,@function
_Z1fv:                                  # @_Z1fv
.Lfunc_begin0:
	.file	0 "/tmp" "frame-types.cpp"
	.loc	0 10 0                          # frame-types.cpp:10:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset %ebp, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register %ebp
.Ltmp0:
	pushl	%ebx
	subl	$372, %esp                      # imm = 0x174
	.cfi_offset %ebx, -12
	.loc	0 13 9 prologue_end             # frame-types.cpp:13:9
	calll	.L0$pb
.L0$pb:
	popl	%ebx
.Ltmp1:
	addl	$_GLOBAL_OFFSET_TABLE_+(.Ltmp1-.L0$pb), %ebx
	leal	-5(%ebp), %eax
	movl	%eax, -16(%ebp)
	.loc	0 14 14                         # frame-types.cpp:14:14
	movb	$1, -21(%ebp)
	.loc	0 14 10 is_stmt 0               # frame-types.cpp:14:10
	leal	-21(%ebp), %eax
	movl	%eax, -20(%ebp)
	.loc	0 17 14 is_stmt 1               # frame-types.cpp:17:14
	movb	$2, -37(%ebp)
	.loc	0 24 14                         # frame-types.cpp:24:14
	leal	-352(%ebp), %eax
	xorl	%ecx, %ecx
	movl	%eax, (%esp)
	movl	$0, 4(%esp)
	movl	$4, 8(%esp)
	calll	memset@PLT
	.loc	0 25 1 epilogue_begin           # frame-types.cpp:25:1
	addl	$372, %esp                      # imm = 0x174
	popl	%ebx
	popl	%ebp
	.cfi_def_cfa %esp, 4
	retl
.Ltmp2:
.Lfunc_end0:
	.size	_Z1fv, .Lfunc_end0-_Z1fv
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	22                              # DW_TAG_typedef
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	16                              # DW_TAG_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	66                              # DW_TAG_rvalue_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	31                              # DW_TAG_ptr_to_member_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	29                              # DW_AT_containing_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
	.byte	55                              # DW_TAG_restrict_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	16                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	17                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	18                              # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	19                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	20                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	21                              # Abbreviation Code
	.byte	13                              # DW_TAG_member
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	56                              # DW_AT_data_member_location
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	4                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x148 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	26                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0xa6 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	85
	.byte	3                               # DW_AT_linkage_name
	.byte	4                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x2f:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	123
	.byte	5                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	201                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x3a:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	7                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	205                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x45:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	8                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	13                              # DW_AT_decl_line
	.long	210                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x50:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	108
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.long	215                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x5b:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	96
	.byte	10                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	15                              # DW_AT_decl_line
	.long	220                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x66:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	92
	.byte	4                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
	.long	247                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x71:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	91
	.byte	12                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	17                              # DW_AT_decl_line
	.long	256                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x7c:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	90
	.byte	13                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.long	261                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x87:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	84
	.byte	14                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.long	266                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x92:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	83
	.byte	15                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	21                              # DW_AT_decl_line
	.long	192                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x9d:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	71
	.byte	17                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	22                              # DW_AT_decl_line
	.long	271                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0xa8:0xc DW_TAG_variable
	.byte	3                               # DW_AT_location
	.byte	145
	.ascii	"\247}"
	.byte	19                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
	.long	287                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0xb4:0xc DW_TAG_variable
	.byte	3                               # DW_AT_location
	.byte	145
	.ascii	"\240}"
	.byte	20                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	24                              # DW_AT_decl_line
	.long	305                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0xc0:0x8 DW_TAG_typedef
	.long	201                             # DW_AT_type
	.byte	16                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	20                              # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0xc9:0x4 DW_TAG_base_type
	.byte	6                               # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0xcd:0x5 DW_TAG_pointer_type
	.long	201                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0xd2:0x5 DW_TAG_reference_type
	.long	201                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0xd7:0x5 DW_TAG_rvalue_reference_type
	.long	201                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0xdc:0x9 DW_TAG_ptr_to_member_type
	.long	229                             # DW_AT_type
	.long	245                             # DW_AT_containing_type
	.byte	10                              # Abbrev [10] 0xe5:0xb DW_TAG_subroutine_type
	.long	201                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0xea:0x5 DW_TAG_formal_parameter
	.long	240                             # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0xf0:0x5 DW_TAG_pointer_type
	.long	245                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0xf5:0x2 DW_TAG_structure_type
	.byte	11                              # DW_AT_name
                                        # DW_AT_declaration
	.byte	9                               # Abbrev [9] 0xf7:0x9 DW_TAG_ptr_to_member_type
	.long	201                             # DW_AT_type
	.long	245                             # DW_AT_containing_type
	.byte	13                              # Abbrev [13] 0x100:0x5 DW_TAG_const_type
	.long	201                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0x105:0x5 DW_TAG_volatile_type
	.long	201                             # DW_AT_type
	.byte	15                              # Abbrev [15] 0x10a:0x5 DW_TAG_restrict_type
	.long	205                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x10f:0xc DW_TAG_array_type
	.long	201                             # DW_AT_type
	.byte	17                              # Abbrev [17] 0x114:0x6 DW_TAG_subrange_type
	.long	283                             # DW_AT_type
	.byte	12                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0x11b:0x4 DW_TAG_base_type
	.byte	18                              # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	16                              # Abbrev [16] 0x11f:0x12 DW_TAG_array_type
	.long	201                             # DW_AT_type
	.byte	17                              # Abbrev [17] 0x124:0x6 DW_TAG_subrange_type
	.long	283                             # DW_AT_type
	.byte	12                              # DW_AT_count
	.byte	17                              # Abbrev [17] 0x12a:0x6 DW_TAG_subrange_type
	.long	283                             # DW_AT_type
	.byte	24                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x131:0x8 DW_TAG_typedef
	.long	313                             # DW_AT_type
	.byte	24                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	19                              # Abbrev [19] 0x139:0x16 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	23                              # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	20                              # Abbrev [20] 0x13f:0x6 DW_TAG_template_type_parameter
	.long	335                             # DW_AT_type
	.byte	22                              # DW_AT_name
	.byte	21                              # Abbrev [21] 0x145:0x9 DW_TAG_member
	.byte	20                              # DW_AT_name
	.long	335                             # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x14f:0x4 DW_TAG_base_type
	.byte	21                              # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	104                             # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git" # string offset=0
.Linfo_string1:
	.asciz	"frame-types.cpp"               # string offset=107
.Linfo_string2:
	.asciz	"/tmp"                          # string offset=123
.Linfo_string3:
	.asciz	"_Z1fv"                         # string offset=128
.Linfo_string4:
	.asciz	"f"                             # string offset=134
.Linfo_string5:
	.asciz	"a"                             # string offset=136
.Linfo_string6:
	.asciz	"char"                          # string offset=138
.Linfo_string7:
	.asciz	"b"                             # string offset=143
.Linfo_string8:
	.asciz	"c"                             # string offset=145
.Linfo_string9:
	.asciz	"d"                             # string offset=147
.Linfo_string10:
	.asciz	"e"                             # string offset=149
.Linfo_string11:
	.asciz	"S"                             # string offset=151
.Linfo_string12:
	.asciz	"g"                             # string offset=153
.Linfo_string13:
	.asciz	"h"                             # string offset=155
.Linfo_string14:
	.asciz	"i"                             # string offset=157
.Linfo_string15:
	.asciz	"j"                             # string offset=159
.Linfo_string16:
	.asciz	"char_typedef"                  # string offset=161
.Linfo_string17:
	.asciz	"k"                             # string offset=174
.Linfo_string18:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=176
.Linfo_string19:
	.asciz	"l"                             # string offset=196
.Linfo_string20:
	.asciz	"m"                             # string offset=198
.Linfo_string21:
	.asciz	"int"                           # string offset=200
.Linfo_string22:
	.asciz	"Y"                             # string offset=204
.Linfo_string23:
	.asciz	"Base<int>"                     # string offset=206
.Linfo_string24:
	.asciz	"Alias<int>"                    # string offset=216
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.long	.Linfo_string13
	.long	.Linfo_string14
	.long	.Linfo_string15
	.long	.Linfo_string16
	.long	.Linfo_string17
	.long	.Linfo_string18
	.long	.Linfo_string19
	.long	.Linfo_string20
	.long	.Linfo_string21
	.long	.Linfo_string22
	.long	.Linfo_string23
	.long	.Linfo_string24
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	4                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.long	.Lfunc_begin0
.Ldebug_addr_end0:
	.ident	"clang version 19.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
