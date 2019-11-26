// REQUIRES: aarch64-registered-target

// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android -o %t.o %s
// RUN: echo 'FRAME %t.o 4' | llvm-symbolizer | FileCheck %s

// CHECK:      f
// CHECK-NEXT: this
// CHECK-NEXT: ??:0
// CHECK-NEXT: ?? 8 ??
// CHECK-NEXT: f
// CHECK-NEXT: x
// CHECK-NEXT: {{.*}}dbg.cc:8
// CHECK-NEXT: -4 4 ??

	.text
	.file	"dbg.cc"
	.file	1 "/tmp" "dbg.cc"
	.globl	_ZN1A1fEv               // -- Begin function _ZN1A1fEv
	.p2align	2
	.type	_ZN1A1fEv,@function
_ZN1A1fEv:                              // @_ZN1A1fEv
.Lfunc_begin0:
	.loc	1 7 0                   // /tmp/dbg.cc:7:0
	.cfi_startproc
// %bb.0:                               // %entry
	//DEBUG_VALUE: f:this <- $x0
	sub	sp, sp, #32             // =32
	stp	x29, x30, [sp, #16]     // 16-byte Folded Spill
	add	x29, sp, #16            // =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
.Ltmp0:
	//DEBUG_VALUE: f:x <- [DW_OP_constu 4, DW_OP_minus, DW_OP_deref] $fp
	.loc	1 9 3 prologue_end      // /tmp/dbg.cc:9:3
	sub	x0, x29, #4             // =4
.Ltmp1:
	bl	_Z3usePi
.Ltmp2:
	.loc	1 10 1                  // /tmp/dbg.cc:10:1
	ldp	x29, x30, [sp, #16]     // 16-byte Folded Reload
	add	sp, sp, #32             // =32
	ret
.Ltmp3:
.Lfunc_end0:
	.size	_ZN1A1fEv, .Lfunc_end0-_ZN1A1fEv
	.cfi_endproc
                                        // -- End function
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 5d25e153457e7bdd4181ab7496402a565a5b9370)" // string offset=0
.Linfo_string1:
	.asciz	"/tmp/dbg.cc"           // string offset=101
.Linfo_string2:
	.asciz	"/code/build-llvm-cmake" // string offset=113
.Linfo_string3:
	.asciz	"_Z3usePi"              // string offset=136
.Linfo_string4:
	.asciz	"use"                   // string offset=145
.Linfo_string5:
	.asciz	"int"                   // string offset=149
.Linfo_string6:
	.asciz	"_ZN1A1fEv"             // string offset=153
.Linfo_string7:
	.asciz	"f"                     // string offset=163
.Linfo_string8:
	.asciz	"A"                     // string offset=165
.Linfo_string9:
	.asciz	"this"                  // string offset=167
.Linfo_string10:
	.asciz	"x"                     // string offset=172
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.xword	.Lfunc_begin0-.Lfunc_begin0
	.xword	.Ltmp1-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	80                      // DW_OP_reg0
	.xword	0
	.xword	0
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
	.byte	3                       // Abbreviation Code
	.byte	5                       // DW_TAG_formal_parameter
	.byte	0                       // DW_CHILDREN_no
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	4                       // Abbreviation Code
	.byte	15                      // DW_TAG_pointer_type
	.byte	0                       // DW_CHILDREN_no
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	5                       // Abbreviation Code
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
	.byte	6                       // Abbreviation Code
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
	.byte	7                       // Abbreviation Code
	.byte	5                       // DW_TAG_formal_parameter
	.byte	0                       // DW_CHILDREN_no
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	52                      // DW_AT_artificial
	.byte	25                      // DW_FORM_flag_present
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	8                       // Abbreviation Code
	.byte	46                      // DW_TAG_subprogram
	.byte	1                       // DW_CHILDREN_yes
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
	.byte	18                      // DW_AT_high_pc
	.byte	6                       // DW_FORM_data4
	.byte	64                      // DW_AT_frame_base
	.byte	24                      // DW_FORM_exprloc
	.byte	100                     // DW_AT_object_pointer
	.byte	19                      // DW_FORM_ref4
	.ascii	"\227B"                 // DW_AT_GNU_all_call_sites
	.byte	25                      // DW_FORM_flag_present
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	71                      // DW_AT_specification
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	9                       // Abbreviation Code
	.byte	5                       // DW_TAG_formal_parameter
	.byte	0                       // DW_CHILDREN_no
	.byte	2                       // DW_AT_location
	.byte	23                      // DW_FORM_sec_offset
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	52                      // DW_AT_artificial
	.byte	25                      // DW_FORM_flag_present
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	10                      // Abbreviation Code
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
	.byte	11                      // Abbreviation Code
	.ascii	"\211\202\001"          // DW_TAG_GNU_call_site
	.byte	0                       // DW_CHILDREN_no
	.byte	49                      // DW_AT_abstract_origin
	.byte	19                      // DW_FORM_ref4
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
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
	.byte	1                       // Abbrev [1] 0xb:0xa3 DW_TAG_compile_unit
	.word	.Linfo_string0          // DW_AT_producer
	.hword	33                      // DW_AT_language
	.word	.Linfo_string1          // DW_AT_name
	.word	.Lline_table_start0     // DW_AT_stmt_list
	.word	.Linfo_string2          // DW_AT_comp_dir
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.byte	2                       // Abbrev [2] 0x2a:0x11 DW_TAG_subprogram
	.word	.Linfo_string3          // DW_AT_linkage_name
	.word	.Linfo_string4          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	1                       // DW_AT_decl_line
                                        // DW_AT_declaration
                                        // DW_AT_external
	.byte	3                       // Abbrev [3] 0x35:0x5 DW_TAG_formal_parameter
	.word	59                      // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	4                       // Abbrev [4] 0x3b:0x5 DW_TAG_pointer_type
	.word	64                      // DW_AT_type
	.byte	5                       // Abbrev [5] 0x40:0x7 DW_TAG_base_type
	.word	.Linfo_string5          // DW_AT_name
	.byte	5                       // DW_AT_encoding
	.byte	4                       // DW_AT_byte_size
	.byte	6                       // Abbrev [6] 0x47:0x1b DW_TAG_class_type
	.byte	5                       // DW_AT_calling_convention
	.word	.Linfo_string8          // DW_AT_name
	.byte	1                       // DW_AT_byte_size
	.byte	1                       // DW_AT_decl_file
	.byte	3                       // DW_AT_decl_line
	.byte	2                       // Abbrev [2] 0x50:0x11 DW_TAG_subprogram
	.word	.Linfo_string6          // DW_AT_linkage_name
	.word	.Linfo_string7          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	4                       // DW_AT_decl_line
                                        // DW_AT_declaration
                                        // DW_AT_external
	.byte	7                       // Abbrev [7] 0x5b:0x5 DW_TAG_formal_parameter
	.word	98                      // DW_AT_type
                                        // DW_AT_artificial
	.byte	0                       // End Of Children Mark
	.byte	0                       // End Of Children Mark
	.byte	4                       // Abbrev [4] 0x62:0x5 DW_TAG_pointer_type
	.word	71                      // DW_AT_type
	.byte	8                       // Abbrev [8] 0x67:0x41 DW_TAG_subprogram
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
	.word	127                     // DW_AT_object_pointer
                                        // DW_AT_GNU_all_call_sites
	.byte	7                       // DW_AT_decl_line
	.word	80                      // DW_AT_specification
	.byte	9                       // Abbrev [9] 0x7f:0xd DW_TAG_formal_parameter
	.word	.Ldebug_loc0            // DW_AT_location
	.word	.Linfo_string9          // DW_AT_name
	.word	168                     // DW_AT_type
                                        // DW_AT_artificial
	.byte	10                      // Abbrev [10] 0x8c:0xe DW_TAG_variable
	.byte	2                       // DW_AT_location
	.byte	145
	.byte	124
	.word	.Linfo_string10         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	8                       // DW_AT_decl_line
	.word	64                      // DW_AT_type
	.byte	11                      // Abbrev [11] 0x9a:0xd DW_TAG_GNU_call_site
	.word	42                      // DW_AT_abstract_origin
	.xword	.Ltmp2                  // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	4                       // Abbrev [4] 0xa8:0x5 DW_TAG_pointer_type
	.word	71                      // DW_AT_type
	.byte	0                       // End Of Children Mark
.Ldebug_info_end0:
	.ident	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 5d25e153457e7bdd4181ab7496402a565a5b9370)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
