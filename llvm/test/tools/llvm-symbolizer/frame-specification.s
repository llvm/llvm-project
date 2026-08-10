// REQUIRES: aarch64-registered-target

// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android -o %t.o %s
// RUN: echo 'FRAME %t.o 4' | llvm-symbolizer | FileCheck %s

// Built from the following source with
// clang -target aarch64-linux-android -O3 -g -S
//
// class A {
//   static void f();
// };
//
// void use(int*);
// void A::f() {int x; use(&x);}

// CHECK:      {{^f$}}
// CHECK-NEXT: {{^x$}}
// CHECK-NEXT: {{.*}}dbg.cc:6
// CHECK-NEXT: -4 4 ??

	.text
	.file	"dbg.cc"
	.globl	_ZN1A1fEv               // -- Begin function _ZN1A1fEv
	.p2align	2
	.type	_ZN1A1fEv,@function
_ZN1A1fEv:                              // @_ZN1A1fEv
.Lfunc_begin0:
	.file	1 "/tmp" "dbg.cc"
	.loc	1 6 0                   // /tmp/dbg.cc:6:0
	.cfi_startproc
// %bb.0:                               // %entry
	sub	sp, sp, #32             // =32
	stp	x29, x30, [sp, #16]     // 16-byte Folded Spill
	add	x29, sp, #16            // =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
.Ltmp0:
	//DEBUG_VALUE: f:x <- [DW_OP_constu 4, DW_OP_minus, DW_OP_deref] $fp
	.loc	1 6 21 prologue_end     // /tmp/dbg.cc:6:21
	sub	x0, x29, #4             // =4
	bl	_Z3usePi
.Ltmp1:
	.loc	1 6 29 is_stmt 0        // /tmp/dbg.cc:6:29
	ldp	x29, x30, [sp, #16]     // 16-byte Folded Reload
	add	sp, sp, #32             // =32
	ret
.Ltmp2:
.Lfunc_end0:
	.size	_ZN1A1fEv, .Lfunc_end0-_ZN1A1fEv
	.cfi_endproc
                                        // -- End function
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 092ef9c6cf4678d2b8df7738110ecd405fe1fe3d)" // string offset=0
.Linfo_string1:
	.asciz	"/tmp/dbg.cc"           // string offset=101
.Linfo_string2:
	.asciz	"/code/build-llvm-cmake" // string offset=113
.Linfo_string3:
	.asciz	"_ZN1A1fEv"             // string offset=136
.Linfo_string4:
	.asciz	"f"                     // string offset=146
.Linfo_string5:
	.asciz	"A"                     // string offset=148
.Linfo_string6:
	.asciz	"_Z3usePi"              // string offset=150
.Linfo_string7:
	.asciz	"use"                   // string offset=159
.Linfo_string8:
	.asciz	"int"                   // string offset=163
.Linfo_string9:
	.asciz	"x"                     // string offset=167
	.section	.debug_abbrev,"",@progbits
	.byte	1                       // Abbreviation Code
	.byte	17                      // DW_TAG_compile_unit
	.byte	1                       // DW_CHILDREN_yes
	.byte	37                      // DW_AT_producer
	.byte	14                      // DW_FORM_strp
	.byte	19                      // DW_AT_language
	.byte	5                       // DW_FORM_data2
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	16                      // DW_AT_stmt_list
	.byte	23                      // DW_FORM_sec_offset
	.byte	27                      // DW_AT_comp_dir
	.byte	14                      // DW_FORM_strp
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
	.byte	18                      // DW_AT_high_pc
	.byte	6                       // DW_FORM_data4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	2                       // Abbreviation Code
	.byte	2                       // DW_TAG_class_type
	.byte	1                       // DW_CHILDREN_yes
	.byte	54                      // DW_AT_calling_convention
	.byte	11                      // DW_FORM_data1
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	11                      // DW_AT_byte_size
	.byte	11                      // DW_FORM_data1
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	3                       // Abbreviation Code
	.byte	46                      // DW_TAG_subprogram
	.byte	0                       // DW_CHILDREN_no
	.byte	110                     // DW_AT_linkage_name
	.byte	14                      // DW_FORM_strp
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	60                      // DW_AT_declaration
	.byte	25                      // DW_FORM_flag_present
	.byte	63                      // DW_AT_external
	.byte	25                      // DW_FORM_flag_present
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	4                       // Abbreviation Code
	.byte	46                      // DW_TAG_subprogram
	.byte	1                       // DW_CHILDREN_yes
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
	.byte	18                      // DW_AT_high_pc
	.byte	6                       // DW_FORM_data4
	.byte	64                      // DW_AT_frame_base
	.byte	24                      // DW_FORM_exprloc
	.ascii	"\227B"                 // DW_AT_GNU_all_call_sites
	.byte	25                      // DW_FORM_flag_present
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	71                      // DW_AT_specification
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	5                       // Abbreviation Code
	.byte	52                      // DW_TAG_variable
	.byte	0                       // DW_CHILDREN_no
	.byte	2                       // DW_AT_location
	.byte	24                      // DW_FORM_exprloc
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	6                       // Abbreviation Code
	.ascii	"\211\202\001"          // DW_TAG_GNU_call_site
	.byte	0                       // DW_CHILDREN_no
	.byte	49                      // DW_AT_abstract_origin
	.byte	19                      // DW_FORM_ref4
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	7                       // Abbreviation Code
	.byte	46                      // DW_TAG_subprogram
	.byte	1                       // DW_CHILDREN_yes
	.byte	110                     // DW_AT_linkage_name
	.byte	14                      // DW_FORM_strp
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	60                      // DW_AT_declaration
	.byte	25                      // DW_FORM_flag_present
	.byte	63                      // DW_AT_external
	.byte	25                      // DW_FORM_flag_present
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	8                       // Abbreviation Code
	.byte	5                       // DW_TAG_formal_parameter
	.byte	0                       // DW_CHILDREN_no
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	9                       // Abbreviation Code
	.byte	15                      // DW_TAG_pointer_type
	.byte	0                       // DW_CHILDREN_no
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	10                      // Abbreviation Code
	.byte	36                      // DW_TAG_base_type
	.byte	0                       // DW_CHILDREN_no
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	62                      // DW_AT_encoding
	.byte	11                      // DW_FORM_data1
	.byte	11                      // DW_AT_byte_size
	.byte	11                      // DW_FORM_data1
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	0                       // EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	.Ldebug_info_end0-.Ldebug_info_start0 // Length of Unit
.Ldebug_info_start0:
	.hword	4                       // DWARF version number
	.word	.debug_abbrev           // Offset Into Abbrev. Section
	.byte	8                       // Address Size (in bytes)
	.byte	1                       // Abbrev [1] 0xb:0x82 DW_TAG_compile_unit
	.word	.Linfo_string0          // DW_AT_producer
	.hword	33                      // DW_AT_language
	.word	.Linfo_string1          // DW_AT_name
	.word	.Lline_table_start0     // DW_AT_stmt_list
	.word	.Linfo_string2          // DW_AT_comp_dir
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.byte	2                       // Abbrev [2] 0x2a:0x15 DW_TAG_class_type
	.byte	5                       // DW_AT_calling_convention
	.word	.Linfo_string5          // DW_AT_name
	.byte	1                       // DW_AT_byte_size
	.byte	1                       // DW_AT_decl_file
	.byte	1                       // DW_AT_decl_line
	.byte	3                       // Abbrev [3] 0x33:0xb DW_TAG_subprogram
	.word	.Linfo_string3          // DW_AT_linkage_name
	.word	.Linfo_string4          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	2                       // DW_AT_decl_line
                                        // DW_AT_declaration
                                        // DW_AT_external
	.byte	0                       // End Of Children Mark
	.byte	4                       // Abbrev [4] 0x3f:0x30 DW_TAG_subprogram
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_GNU_all_call_sites
	.byte	6                       // DW_AT_decl_line
	.word	51                      // DW_AT_specification
	.byte	5                       // Abbrev [5] 0x53:0xe DW_TAG_variable
	.byte	2                       // DW_AT_location
	.byte	145
	.byte	124
	.word	.Linfo_string9          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	6                       // DW_AT_decl_line
	.word	133                     // DW_AT_type
	.byte	6                       // Abbrev [6] 0x61:0xd DW_TAG_GNU_call_site
	.word	111                     // DW_AT_abstract_origin
	.xword	.Ltmp1                  // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	7                       // Abbrev [7] 0x6f:0x11 DW_TAG_subprogram
	.word	.Linfo_string6          // DW_AT_linkage_name
	.word	.Linfo_string7          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	5                       // DW_AT_decl_line
                                        // DW_AT_declaration
                                        // DW_AT_external
	.byte	8                       // Abbrev [8] 0x7a:0x5 DW_TAG_formal_parameter
	.word	128                     // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	9                       // Abbrev [9] 0x80:0x5 DW_TAG_pointer_type
	.word	133                     // DW_AT_type
	.byte	10                      // Abbrev [10] 0x85:0x7 DW_TAG_base_type
	.word	.Linfo_string8          // DW_AT_name
	.byte	5                       // DW_AT_encoding
	.byte	4                       // DW_AT_byte_size
	.byte	0                       // End Of Children Mark
.Ldebug_info_end0:
	.ident	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 092ef9c6cf4678d2b8df7738110ecd405fe1fe3d)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
