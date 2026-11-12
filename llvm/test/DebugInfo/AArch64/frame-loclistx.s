// Test dwarf-5 DW_AT_location [DW_FORM_loclistx].

// Built with "clang -c -target aarch64-linux -gdwarf-5 -O3 -S" from the following source:
// void use(int *);
// void f(void) {
//   int x = 1;
//   use(&x);
// }

// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android --dwarf-version=5 -o %t.o %s
// RUN: echo 'FRAME %t.o 0x4'  | llvm-symbolizer | FileCheck %s

// DW_AT_location        (indexed (0x0) loclist = 0x00000010:
// [0x0000000000000010, 0x0000000000000018): DW_OP_consts +1, DW_OP_stack_value
// [0x0000000000000018, 0x0000000000000028): DW_OP_breg29 W29-4)
// CHECK:      f
// CHECK-NEXT: x
// CHECK-NEXT: {{.*}}dbg.c:3
// CHECK-NEXT: -4 4 ??

	.text
	.file	"dbg.c"
	.file	0 "/code/build-llvm-cmake" "/tmp/dbg.c" md5 0x87e53f5ae1de5f2c6ec0baa7fe683192
	.globl	_Z1fv                   // -- Begin function _Z1fv
	.p2align	2
	.type	_Z1fv,@function
_Z1fv:                                  // @_Z1fv
.Lfunc_begin0:
	.loc	0 2 0                   // /tmp/dbg.c:2:0
	.cfi_startproc
// %bb.0:                               // %entry
	sub	sp, sp, #32             // =32
	stp	x29, x30, [sp, #16]     // 16-byte Folded Spill
	add	x29, sp, #16            // =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	w8, #1
.Ltmp0:
	//DEBUG_VALUE: f:x <- 1
	.loc	0 4 3 prologue_end      // /tmp/dbg.c:4:3
	sub	x0, x29, #4             // =4
	.loc	0 3 7                   // /tmp/dbg.c:3:7
	stur	w8, [x29, #-4]
.Ltmp1:
	//DEBUG_VALUE: f:x <- [DW_OP_constu 4, DW_OP_minus, DW_OP_deref] $fp
	.loc	0 4 3                   // /tmp/dbg.c:4:3
	bl	_Z3usePi
.Ltmp2:
	.loc	0 5 1                   // /tmp/dbg.c:5:1
	ldp	x29, x30, [sp, #16]     // 16-byte Folded Reload
	add	sp, sp, #32             // =32
	ret
.Ltmp3:
.Lfunc_end0:
	.size	_Z1fv, .Lfunc_end0-_Z1fv
	.cfi_endproc
                                        // -- End function
	.section	.debug_str_offsets,"",@progbits
	.word	40
	.hword	5
	.hword	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 0a094679914ccea0f9c8f03165132172544d9799)" // string offset=0
.Linfo_string1:
	.asciz	"/tmp/dbg.c"            // string offset=101
.Linfo_string2:
	.asciz	"/code/build-llvm-cmake" // string offset=112
.Linfo_string3:
	.asciz	"_Z3usePi"              // string offset=135
.Linfo_string4:
	.asciz	"use"                   // string offset=144
.Linfo_string5:
	.asciz	"int"                   // string offset=148
.Linfo_string6:
	.asciz	"_Z1fv"                 // string offset=152
.Linfo_string7:
	.asciz	"f"                     // string offset=158
.Linfo_string8:
	.asciz	"x"                     // string offset=160
	.section	.debug_str_offsets,"",@progbits
	.word	.Linfo_string0
	.word	.Linfo_string1
	.word	.Linfo_string2
	.word	.Linfo_string3
	.word	.Linfo_string4
	.word	.Linfo_string5
	.word	.Linfo_string6
	.word	.Linfo_string7
	.word	.Linfo_string8
	.section	.debug_loclists,"",@progbits
	.word	.Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0 // Length
.Ldebug_loclist_table_start0:
	.hword	5                       // Version
	.byte	8                       // Address size
	.byte	0                       // Segment selector size
	.word	1                       // Offset entry count
.Lloclists_table_base0:
	.word	.Ldebug_loc0-.Lloclists_table_base0
.Ldebug_loc0:
	.byte	4                       // DW_LLE_offset_pair
	.uleb128 .Ltmp0-.Lfunc_begin0   //   starting offset
	.uleb128 .Ltmp1-.Lfunc_begin0   //   ending offset
	.byte	3                       // Loc expr size
	.byte	17                      // DW_OP_consts
	.byte	1                       // 1
	.byte	159                     // DW_OP_stack_value
	.byte	4                       // DW_LLE_offset_pair
	.uleb128 .Ltmp1-.Lfunc_begin0   //   starting offset
	.uleb128 .Lfunc_end0-.Lfunc_begin0 //   ending offset
	.byte	2                       // Loc expr size
	.byte	141                     // DW_OP_breg29
	.byte	124                     // -4
	.byte	0                       // DW_LLE_end_of_list
.Ldebug_loclist_table_end0:
	.section	.debug_abbrev,"",@progbits
	.byte	1                       // Abbreviation Code
	.byte	17                      // DW_TAG_compile_unit
	.byte	1                       // DW_CHILDREN_yes
	.byte	37                      // DW_AT_producer
	.byte	37                      // DW_FORM_strx1
	.byte	19                      // DW_AT_language
	.byte	5                       // DW_FORM_data2
	.byte	3                       // DW_AT_name
	.byte	37                      // DW_FORM_strx1
	.byte	114                     // DW_AT_str_offsets_base
	.byte	23                      // DW_FORM_sec_offset
	.byte	16                      // DW_AT_stmt_list
	.byte	23                      // DW_FORM_sec_offset
	.byte	27                      // DW_AT_comp_dir
	.byte	37                      // DW_FORM_strx1
	.byte	17                      // DW_AT_low_pc
	.byte	27                      // DW_FORM_addrx
	.byte	18                      // DW_AT_high_pc
	.byte	6                       // DW_FORM_data4
	.byte	115                     // DW_AT_addr_base
	.byte	23                      // DW_FORM_sec_offset
	.ascii	"\214\001"              // DW_AT_loclists_base
	.byte	23                      // DW_FORM_sec_offset
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	2                       // Abbreviation Code
	.byte	46                      // DW_TAG_subprogram
	.byte	1                       // DW_CHILDREN_yes
	.byte	110                     // DW_AT_linkage_name
	.byte	37                      // DW_FORM_strx1
	.byte	3                       // DW_AT_name
	.byte	37                      // DW_FORM_strx1
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
	.byte	37                      // DW_FORM_strx1
	.byte	62                      // DW_AT_encoding
	.byte	11                      // DW_FORM_data1
	.byte	11                      // DW_AT_byte_size
	.byte	11                      // DW_FORM_data1
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	6                       // Abbreviation Code
	.byte	46                      // DW_TAG_subprogram
	.byte	1                       // DW_CHILDREN_yes
	.byte	17                      // DW_AT_low_pc
	.byte	27                      // DW_FORM_addrx
	.byte	18                      // DW_AT_high_pc
	.byte	6                       // DW_FORM_data4
	.byte	64                      // DW_AT_frame_base
	.byte	24                      // DW_FORM_exprloc
	.byte	122                     // DW_AT_call_all_calls
	.byte	25                      // DW_FORM_flag_present
	.byte	110                     // DW_AT_linkage_name
	.byte	37                      // DW_FORM_strx1
	.byte	3                       // DW_AT_name
	.byte	37                      // DW_FORM_strx1
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	63                      // DW_AT_external
	.byte	25                      // DW_FORM_flag_present
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	7                       // Abbreviation Code
	.byte	52                      // DW_TAG_variable
	.byte	0                       // DW_CHILDREN_no
	.byte	2                       // DW_AT_location
	.byte	34                      // DW_FORM_loclistx
	.byte	3                       // DW_AT_name
	.byte	37                      // DW_FORM_strx1
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	8                       // Abbreviation Code
	.byte	72                      // DW_TAG_call_site
	.byte	0                       // DW_CHILDREN_no
	.byte	127                     // DW_AT_call_origin
	.byte	19                      // DW_FORM_ref4
	.byte	125                     // DW_AT_call_return_pc
	.byte	1                       // DW_FORM_addr
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	0                       // EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	.Ldebug_info_end0-.Ldebug_info_start0 // Length of Unit
.Ldebug_info_start0:
	.hword	5                       // DWARF version number
	.byte	1                       // DWARF Unit Type
	.byte	8                       // Address Size (in bytes)
	.word	.debug_abbrev           // Offset Into Abbrev. Section
	.byte	1                       // Abbrev [1] 0xc:0x53 DW_TAG_compile_unit
	.byte	0                       // DW_AT_producer
	.hword	33                      // DW_AT_language
	.byte	1                       // DW_AT_name
	.word	.Lstr_offsets_base0     // DW_AT_str_offsets_base
	.word	.Lline_table_start0     // DW_AT_stmt_list
	.byte	2                       // DW_AT_comp_dir
	.byte	0                       // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.word	.Laddr_table_base0      // DW_AT_addr_base
	.word	.Lloclists_table_base0  // DW_AT_loclists_base
	.byte	2                       // Abbrev [2] 0x27:0xb DW_TAG_subprogram
	.byte	3                       // DW_AT_linkage_name
	.byte	4                       // DW_AT_name
	.byte	0                       // DW_AT_decl_file
	.byte	1                       // DW_AT_decl_line
                                        // DW_AT_declaration
                                        // DW_AT_external
	.byte	3                       // Abbrev [3] 0x2c:0x5 DW_TAG_formal_parameter
	.word	50                      // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	4                       // Abbrev [4] 0x32:0x5 DW_TAG_pointer_type
	.word	55                      // DW_AT_type
	.byte	5                       // Abbrev [5] 0x37:0x4 DW_TAG_base_type
	.byte	5                       // DW_AT_name
	.byte	5                       // DW_AT_encoding
	.byte	4                       // DW_AT_byte_size
	.byte	6                       // Abbrev [6] 0x3b:0x23 DW_TAG_subprogram
	.byte	0                       // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_call_all_calls
	.byte	6                       // DW_AT_linkage_name
	.byte	7                       // DW_AT_name
	.byte	0                       // DW_AT_decl_file
	.byte	2                       // DW_AT_decl_line
                                        // DW_AT_external
	.byte	7                       // Abbrev [7] 0x47:0x9 DW_TAG_variable
	.byte	0                       // DW_AT_location
	.byte	8                       // DW_AT_name
	.byte	0                       // DW_AT_decl_file
	.byte	3                       // DW_AT_decl_line
	.word	55                      // DW_AT_type
	.byte	8                       // Abbrev [8] 0x50:0xd DW_TAG_call_site
	.word	39                      // DW_AT_call_origin
	.xword	.Ltmp2-.Lfunc_begin0    // DW_AT_call_return_pc
	.byte	0                       // End Of Children Mark
	.byte	0                       // End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_addr,"",@progbits
	.word	.Ldebug_addr_end0-.Ldebug_addr_start0 // Length of contribution
.Ldebug_addr_start0:
	.hword	5                       // DWARF version number
	.byte	8                       // Address size
	.byte	0                       // Segment selector size
.Laddr_table_base0:
	.xword	.Lfunc_begin0
.Ldebug_addr_end0:
	.ident	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 0a094679914ccea0f9c8f03165132172544d9799)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
