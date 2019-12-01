// A hand-crafted object with AT_name fields for the function and the variable commented out.

// REQUIRES: aarch64-registered-target

// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android -o %t.o %s
// RUN: echo 'FRAME %t.o 0' | llvm-symbolizer | FileCheck %s

// CHECK:      ??
// CHECK-NEXT: ??
// CHECK-NEXT: /tmp/dbg.cc:2
// CHECK-NEXT: ?? ?? ??

	.text
	.file	"dbg.cc"
	.globl	func                    // -- Begin function func
	.p2align	2
	.type	func,@function
func:                                   // @func
.Lfunc_begin0:
	.file	1 "/tmp" "dbg.cc"
	.loc	1 1 0                   // /tmp/dbg.cc:1:0
	.cfi_sections .debug_frame
	.cfi_startproc
// %bb.0:                               // %entry
	.loc	1 3 1 prologue_end      // /tmp/dbg.cc:3:1
	ret
.Ltmp0:
.Lfunc_end0:
	.size	func, .Lfunc_end0-func
	.cfi_endproc
                                        // -- End function
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 2102f157f2d1ca6e7e4432b91d75f04e5023172f)" // string offset=0
.Linfo_string1:
	.asciz	"/tmp/dbg.cc"           // string offset=101
.Linfo_string2:
	.asciz	"/code/build-llvm-cmake" // string offset=113
.Linfo_string3:
	.asciz	"func"                  // string offset=136
.Linfo_string4:
	.asciz	"x"                     // string offset=141
.Linfo_string5:
	.asciz	"int"                   // string offset=143
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
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
	.byte	18                      // DW_AT_high_pc
	.byte	6                       // DW_FORM_data4
	.byte	64                      // DW_AT_frame_base
	.byte	24                      // DW_FORM_exprloc
	.ascii	"\227B"                 // DW_AT_GNU_all_call_sites
	.byte	25                      // DW_FORM_flag_present
//	.byte	3                       // DW_AT_name
//	.byte	14                      // DW_FORM_strp
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	63                      // DW_AT_external
	.byte	25                      // DW_FORM_flag_present
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	3                       // Abbreviation Code
	.byte	52                      // DW_TAG_variable
	.byte	0                       // DW_CHILDREN_no
//	.byte	3                       // DW_AT_name
//	.byte	14                      // DW_FORM_strp
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	4                       // Abbreviation Code
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
	.byte	1                       // Abbrev [1] 0xb:0x48 DW_TAG_compile_unit
	.word	.Linfo_string0          // DW_AT_producer
	.hword	33                      // DW_AT_language
	.word	.Linfo_string1          // DW_AT_name
	.word	.Lline_table_start0     // DW_AT_stmt_list
	.word	.Linfo_string2          // DW_AT_comp_dir
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.byte	2                       // Abbrev [2] 0x2a:0x21 DW_TAG_subprogram
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	111
                                        // DW_AT_GNU_all_call_sites
//	.word	.Linfo_string3          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	1                       // DW_AT_decl_line
                                        // DW_AT_external
	.byte	3                       // Abbrev [3] 0x3f:0xb DW_TAG_variable
//	.word	.Linfo_string4          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	2                       // DW_AT_decl_line
	.word	75                      // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	4                       // Abbrev [4] 0x4b:0x7 DW_TAG_base_type
	.word	.Linfo_string5          // DW_AT_name
	.byte	5                       // DW_AT_encoding
	.byte	4                       // DW_AT_byte_size
	.byte	0                       // End Of Children Mark
.Ldebug_info_end0:
	.ident	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 2102f157f2d1ca6e7e4432b91d75f04e5023172f)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
