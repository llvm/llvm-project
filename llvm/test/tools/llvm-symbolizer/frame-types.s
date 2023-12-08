// REQUIRES: x86-registered-target

// RUN: llvm-mc -filetype=obj -triple=i386-linux-gnu -o %t.o %s
// RUN: echo 'FRAME %t.o 0' | llvm-symbolizer | FileCheck %s

// CHECK: f
// CHECK-NEXT: a
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:4
// CHECK-NEXT: -1 1 ??
// CHECK-NEXT: f
// CHECK-NEXT: b
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:5
// CHECK-NEXT: -8 4 ??
// CHECK-NEXT: f
// CHECK-NEXT: c
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:6
// CHECK-NEXT: -12 4 ??
// CHECK-NEXT: f
// CHECK-NEXT: d
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:7
// CHECK-NEXT: -16 4 ??
// CHECK-NEXT: f
// CHECK-NEXT: e
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:8
// CHECK-NEXT: -32 8 ??
// CHECK-NEXT: f
// CHECK-NEXT: f
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:9
// CHECK-NEXT: -36 4 ??
// CHECK-NEXT: f
// CHECK-NEXT: g
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:10
// CHECK-NEXT: -37 1 ??
// CHECK-NEXT: f
// CHECK-NEXT: h
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:11
// CHECK-NEXT: -38 1 ??
// CHECK-NEXT: f
// CHECK-NEXT: i
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:12
// CHECK-NEXT: -44 4 ??
// CHECK-NEXT: f
// CHECK-NEXT: j
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:14
// CHECK-NEXT: -45 1 ??
// CHECK-NEXT: f
// CHECK-NEXT: k
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:15
// CHECK-NEXT: -57 12 ??
// CHECK-NEXT: f
// CHECK-NEXT: l
// CHECK-NEXT: /tmp{{/|\\}}frame-types.cpp:16
// CHECK-NEXT: -345 288 ??

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
	.globl	_Z1fv                   # -- Begin function _Z1fv
	.p2align	4, 0x90
	.type	_Z1fv,@function
_Z1fv:                                  # @_Z1fv
.Lfunc_begin0:
	.file	1 "/tmp" "frame-types.cpp"
	.loc	1 3 0                   # frame-types.cpp:3:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:                                # %entry
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset %ebp, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register %ebp
	subl	$352, %esp              # imm = 0x160
.Ltmp0:
	.loc	1 6 9 prologue_end      # frame-types.cpp:6:9
	leal	-1(%ebp), %eax
.Ltmp1:
	#DEBUG_VALUE: f:a <- [$eax+0]
	movl	%eax, -12(%ebp)
	.loc	1 7 14                  # frame-types.cpp:7:14
	movb	$1, -17(%ebp)
	.loc	1 7 10 is_stmt 0        # frame-types.cpp:7:10
	leal	-17(%ebp), %eax
.Ltmp2:
	movl	%eax, -16(%ebp)
	.loc	1 10 14 is_stmt 1       # frame-types.cpp:10:14
	movb	$2, -37(%ebp)
	.loc	1 17 1                  # frame-types.cpp:17:1
	addl	$352, %esp              # imm = 0x160
	popl	%ebp
	.cfi_def_cfa %esp, 4
	retl
.Ltmp3:
.Lfunc_end0:
	.size	_Z1fv, .Lfunc_end0-_Z1fv
	.cfi_endproc
                                        # -- End function
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 9.0.0 "  # string offset=0
.Linfo_string1:
	.asciz	"frame-types.cpp"       # string offset=21
.Linfo_string2:
	.asciz	"/tmp"                  # string offset=37
.Linfo_string3:
	.asciz	"_Z1fv"                 # string offset=42
.Linfo_string4:
	.asciz	"f"                     # string offset=48
.Linfo_string5:
	.asciz	"a"                     # string offset=50
.Linfo_string6:
	.asciz	"char"                  # string offset=52
.Linfo_string7:
	.asciz	"b"                     # string offset=57
.Linfo_string8:
	.asciz	"c"                     # string offset=59
.Linfo_string9:
	.asciz	"d"                     # string offset=61
.Linfo_string10:
	.asciz	"e"                     # string offset=63
.Linfo_string11:
	.asciz	"S"                     # string offset=65
.Linfo_string12:
	.asciz	"g"                     # string offset=67
.Linfo_string13:
	.asciz	"h"                     # string offset=69
.Linfo_string14:
	.asciz	"i"                     # string offset=71
.Linfo_string15:
	.asciz	"j"                     # string offset=73
.Linfo_string16:
	.asciz	"char_typedef"          # string offset=75
.Linfo_string17:
	.asciz	"k"                     # string offset=88
.Linfo_string18:
	.asciz	"__ARRAY_SIZE_TYPE__"   # string offset=90
.Linfo_string19:
	.asciz	"l"                     # string offset=110
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.byte	110                     # DW_AT_linkage_name
	.byte	14                      # DW_FORM_strp
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	24                      # DW_FORM_exprloc
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	22                      # DW_TAG_typedef
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	5                       # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	6                       # Abbreviation Code
	.byte	15                      # DW_TAG_pointer_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	7                       # Abbreviation Code
	.byte	16                      # DW_TAG_reference_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	8                       # Abbreviation Code
	.byte	66                      # DW_TAG_rvalue_reference_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	9                       # Abbreviation Code
	.byte	31                      # DW_TAG_ptr_to_member_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	29                      # DW_AT_containing_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	10                      # Abbreviation Code
	.byte	21                      # DW_TAG_subroutine_type
	.byte	1                       # DW_CHILDREN_yes
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	11                      # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	52                      # DW_AT_artificial
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	12                      # Abbreviation Code
	.byte	19                      # DW_TAG_structure_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	60                      # DW_AT_declaration
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	13                      # Abbreviation Code
	.byte	38                      # DW_TAG_const_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	14                      # Abbreviation Code
	.byte	53                      # DW_TAG_volatile_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	15                      # Abbreviation Code
	.byte	55                      # DW_TAG_restrict_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	16                      # Abbreviation Code
	.byte	1                       # DW_TAG_array_type
	.byte	1                       # DW_CHILDREN_yes
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	17                      # Abbreviation Code
	.byte	33                      # DW_TAG_subrange_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	55                      # DW_AT_count
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	18                      # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	4                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x157 DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_producer
	.short	4                       # DW_AT_language
	.long	.Linfo_string1          # DW_AT_name
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.long	.Linfo_string2          # DW_AT_comp_dir
	.long	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	2                       # Abbrev [2] 0x26:0xca DW_TAG_subprogram
	.long	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	85
	.long	.Linfo_string3          # DW_AT_linkage_name
	.long	.Linfo_string4          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	3                       # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                       # Abbrev [3] 0x3b:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	127
	.long	.Linfo_string5          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	4                       # DW_AT_decl_line
	.long	240                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0x49:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string7          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	5                       # DW_AT_decl_line
	.long	247                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0x57:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	116
	.long	.Linfo_string8          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	6                       # DW_AT_decl_line
	.long	252                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0x65:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string9          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	7                       # DW_AT_decl_line
	.long	257                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0x73:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	96
	.long	.Linfo_string10         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	8                       # DW_AT_decl_line
	.long	262                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0x81:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	92
	.long	.Linfo_string4          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	9                       # DW_AT_decl_line
	.long	292                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0x8f:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	91
	.long	.Linfo_string12         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	10                      # DW_AT_decl_line
	.long	301                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0x9d:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	90
	.long	.Linfo_string13         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	11                      # DW_AT_decl_line
	.long	306                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0xab:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	84
	.long	.Linfo_string14         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	12                      # DW_AT_decl_line
	.long	311                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0xb9:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	83
	.long	.Linfo_string15         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	14                      # DW_AT_decl_line
	.long	228                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0xc7:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	71
	.long	.Linfo_string17         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	15                      # DW_AT_decl_line
	.long	316                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0xd5:0xf DW_TAG_variable
	.byte	3                       # DW_AT_location
	.byte	145
	.ascii	"\247}"
	.long	.Linfo_string19         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	16                      # DW_AT_decl_line
	.long	335                     # DW_AT_type
	.byte	4                       # Abbrev [4] 0xe4:0xb DW_TAG_typedef
	.long	240                     # DW_AT_type
	.long	.Linfo_string16         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	13                      # DW_AT_decl_line
	.byte	0                       # End Of Children Mark
	.byte	5                       # Abbrev [5] 0xf0:0x7 DW_TAG_base_type
	.long	.Linfo_string6          # DW_AT_name
	.byte	6                       # DW_AT_encoding
	.byte	1                       # DW_AT_byte_size
	.byte	6                       # Abbrev [6] 0xf7:0x5 DW_TAG_pointer_type
	.long	240                     # DW_AT_type
	.byte	7                       # Abbrev [7] 0xfc:0x5 DW_TAG_reference_type
	.long	240                     # DW_AT_type
	.byte	8                       # Abbrev [8] 0x101:0x5 DW_TAG_rvalue_reference_type
	.long	240                     # DW_AT_type
	.byte	9                       # Abbrev [9] 0x106:0x9 DW_TAG_ptr_to_member_type
	.long	271                     # DW_AT_type
	.long	287                     # DW_AT_containing_type
	.byte	10                      # Abbrev [10] 0x10f:0xb DW_TAG_subroutine_type
	.long	240                     # DW_AT_type
	.byte	11                      # Abbrev [11] 0x114:0x5 DW_TAG_formal_parameter
	.long	282                     # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                       # End Of Children Mark
	.byte	6                       # Abbrev [6] 0x11a:0x5 DW_TAG_pointer_type
	.long	287                     # DW_AT_type
	.byte	12                      # Abbrev [12] 0x11f:0x5 DW_TAG_structure_type
	.long	.Linfo_string11         # DW_AT_name
                                        # DW_AT_declaration
	.byte	9                       # Abbrev [9] 0x124:0x9 DW_TAG_ptr_to_member_type
	.long	240                     # DW_AT_type
	.long	287                     # DW_AT_containing_type
	.byte	13                      # Abbrev [13] 0x12d:0x5 DW_TAG_const_type
	.long	240                     # DW_AT_type
	.byte	14                      # Abbrev [14] 0x132:0x5 DW_TAG_volatile_type
	.long	240                     # DW_AT_type
	.byte	15                      # Abbrev [15] 0x137:0x5 DW_TAG_restrict_type
	.long	247                     # DW_AT_type
	.byte	16                      # Abbrev [16] 0x13c:0xc DW_TAG_array_type
	.long	240                     # DW_AT_type
	.byte	17                      # Abbrev [17] 0x141:0x6 DW_TAG_subrange_type
	.long	328                     # DW_AT_type
	.byte	12                      # DW_AT_count
	.byte	0                       # End Of Children Mark
	.byte	18                      # Abbrev [18] 0x148:0x7 DW_TAG_base_type
	.long	.Linfo_string18         # DW_AT_name
	.byte	8                       # DW_AT_byte_size
	.byte	7                       # DW_AT_encoding
	.byte	16                      # Abbrev [16] 0x14f:0x12 DW_TAG_array_type
	.long	240                     # DW_AT_type
	.byte	17                      # Abbrev [17] 0x154:0x6 DW_TAG_subrange_type
	.long	328                     # DW_AT_type
	.byte	12                      # DW_AT_count
	.byte	17                      # Abbrev [17] 0x15a:0x6 DW_TAG_subrange_type
	.long	328                     # DW_AT_type
	.byte	24                      # DW_AT_count
	.byte	0                       # End Of Children Mark
	.byte	0                       # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_macinfo,"",@progbits
	.byte	0                       # End Of Macro List Mark

	.ident	"clang version 9.0.0 "
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
