// Test various single-location and location list formats.
// REQUIRES: aarch64-registered-target

// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android -o %t.o %s
// RUN: echo -e 'FRAME %t.o 0x4\nFRAME %t.o 0x24\nFRAME %t.o 0x48\nFRAME %t.o 0x6a\nFRAME %t.o 0x9a' | llvm-symbolizer | FileCheck %s

// DW_AT_location        (DW_OP_breg29 W29-4)
// CHECK:      func00
// CHECK-NEXT: x
// CHECK-NEXT: /tmp/dbg.cc:7
// CHECK-NEXT: -4 4 ??

// DW_AT_location        (DW_OP_fbreg -4)
// CHECK:      func0{{$}}
// CHECK-NEXT: x
// CHECK-NEXT: /tmp/dbg.cc:12
// CHECK-NEXT: -4 4 ??

// DW_AT_location        (0x00000000: 
//   [0x000000000000004c, 0x0000000000000058): DW_OP_breg29 W29-4
//   [0x0000000000000058, 0x000000000000005c): DW_OP_reg0 W0)
// CHECK:      func1
// CHECK-NEXT: x
// CHECK-NEXT: /tmp/dbg.cc:17
// CHECK-NEXT: -4 4 ??

// DW_AT_location        (0x00000037: 
//    [0x0000000000000078, 0x0000000000000080): DW_OP_consts +1, DW_OP_stack_value
//    [0x0000000000000080, 0x0000000000000088): DW_OP_breg29 W29-4
//    [0x0000000000000088, 0x000000000000008c): DW_OP_reg0 W0)
// CHECK:      func2
// CHECK-NEXT: x
// CHECK-NEXT: /tmp/dbg.cc:23
// CHECK-NEXT: -4 4 ??

// location lost
// DW_AT_location        (0x00000083: 
//    [0x00000000000000a8, 0x00000000000000b8): DW_OP_consts +1, DW_OP_stack_value
//    [0x00000000000000b8, 0x00000000000000bc): DW_OP_reg0 W0)
// CHECK:      func3
// CHECK-NEXT: x
// CHECK-NEXT: /tmp/dbg.cc:29
// CHECK-NEXT: ?? 4 ??

	.text
	.file	"dbg.cc"
	.file	1 "/tmp" "dbg.cc"
	.globl	func00                  // -- Begin function func00
	.p2align	2
	.type	func00,@function
func00:                                 // @func00
.Lfunc_begin0:
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
	//DEBUG_VALUE: func00:x <- [DW_OP_constu 4, DW_OP_minus, DW_OP_deref] $fp
	.loc	1 8 3 prologue_end      // /tmp/dbg.cc:8:3
	sub	x0, x29, #4             // =4
	bl	use
.Ltmp1:
	.loc	1 9 1                   // /tmp/dbg.cc:9:1
	ldp	x29, x30, [sp, #16]     // 16-byte Folded Reload
	add	sp, sp, #32             // =32
	ret
.Ltmp2:
.Lfunc_end0:
	.size	func00, .Lfunc_end0-func00
	.cfi_endproc
                                        // -- End function
	.globl	func0                   // -- Begin function func0
	.p2align	2
	.type	func0,@function
func0:                                  // @func0
.Lfunc_begin1:
	.loc	1 11 0                  // /tmp/dbg.cc:11:0
	.cfi_startproc
// %bb.0:                               // %entry
	sub	sp, sp, #32             // =32
	stp	x29, x30, [sp, #16]     // 16-byte Folded Spill
	add	x29, sp, #16            // =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
.Ltmp3:
	//DEBUG_VALUE: func0:x <- [DW_OP_constu 4, DW_OP_minus, DW_OP_deref] $fp
	.loc	1 13 3 prologue_end     // /tmp/dbg.cc:13:3
	sub	x0, x29, #4             // =4
	bl	use
.Ltmp4:
	.loc	1 14 1                  // /tmp/dbg.cc:14:1
	ldp	x29, x30, [sp, #16]     // 16-byte Folded Reload
	add	sp, sp, #32             // =32
	ret
.Ltmp5:
.Lfunc_end1:
	.size	func0, .Lfunc_end1-func0
	.cfi_endproc
                                        // -- End function
	.globl	func1                   // -- Begin function func1
	.p2align	2
	.type	func1,@function
func1:                                  // @func1
.Lfunc_begin2:
	.loc	1 16 0                  // /tmp/dbg.cc:16:0
	.cfi_startproc
// %bb.0:                               // %entry
	sub	sp, sp, #32             // =32
	stp	x29, x30, [sp, #16]     // 16-byte Folded Spill
	add	x29, sp, #16            // =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
.Ltmp6:
	//DEBUG_VALUE: func1:x <- [DW_OP_constu 4, DW_OP_minus, DW_OP_deref] $fp
	.loc	1 18 3 prologue_end     // /tmp/dbg.cc:18:3
	sub	x0, x29, #4             // =4
	bl	use
.Ltmp7:
	.loc	1 19 8                  // /tmp/dbg.cc:19:8
	ldur	w0, [x29, #-4]
.Ltmp8:
	//DEBUG_VALUE: func1:x <- $w0
	.loc	1 19 3 is_stmt 0        // /tmp/dbg.cc:19:3
	bl	usev
.Ltmp9:
	.loc	1 20 1 is_stmt 1        // /tmp/dbg.cc:20:1
	ldp	x29, x30, [sp, #16]     // 16-byte Folded Reload
	add	sp, sp, #32             // =32
	ret
.Ltmp10:
.Lfunc_end2:
	.size	func1, .Lfunc_end2-func1
	.cfi_endproc
                                        // -- End function
	.globl	func2                   // -- Begin function func2
	.p2align	2
	.type	func2,@function
func2:                                  // @func2
.Lfunc_begin3:
	.loc	1 22 0                  // /tmp/dbg.cc:22:0
	.cfi_startproc
// %bb.0:                               // %entry
	sub	sp, sp, #32             // =32
	stp	x29, x30, [sp, #16]     // 16-byte Folded Spill
	add	x29, sp, #16            // =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	w8, #1
.Ltmp11:
	//DEBUG_VALUE: func2:x <- 1
	.loc	1 24 3 prologue_end     // /tmp/dbg.cc:24:3
	sub	x0, x29, #4             // =4
	.loc	1 23 7                  // /tmp/dbg.cc:23:7
	stur	w8, [x29, #-4]
.Ltmp12:
	//DEBUG_VALUE: func2:x <- [DW_OP_constu 4, DW_OP_minus, DW_OP_deref] $fp
	.loc	1 24 3                  // /tmp/dbg.cc:24:3
	bl	use
.Ltmp13:
	.loc	1 25 8                  // /tmp/dbg.cc:25:8
	ldur	w0, [x29, #-4]
.Ltmp14:
	//DEBUG_VALUE: func2:x <- $w0
	.loc	1 25 3 is_stmt 0        // /tmp/dbg.cc:25:3
	bl	usev
.Ltmp15:
	.loc	1 26 1 is_stmt 1        // /tmp/dbg.cc:26:1
	ldp	x29, x30, [sp, #16]     // 16-byte Folded Reload
	add	sp, sp, #32             // =32
	ret
.Ltmp16:
.Lfunc_end3:
	.size	func2, .Lfunc_end3-func2
	.cfi_endproc
                                        // -- End function
	.globl	func3                   // -- Begin function func3
	.p2align	2
	.type	func3,@function
func3:                                  // @func3
.Lfunc_begin4:
	.loc	1 28 0                  // /tmp/dbg.cc:28:0
	.cfi_startproc
// %bb.0:                               // %entry
	sub	sp, sp, #32             // =32
	stp	x29, x30, [sp, #16]     // 16-byte Folded Spill
	add	x29, sp, #16            // =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	w8, #1
.Ltmp17:
	//DEBUG_VALUE: func3:x <- 1
	.loc	1 30 3 prologue_end     // /tmp/dbg.cc:30:3
	sub	x0, x29, #4             // =4
	.loc	1 29 7                  // /tmp/dbg.cc:29:7
	stur	w8, [x29, #-4]
	.loc	1 30 3                  // /tmp/dbg.cc:30:3
	bl	useV
.Ltmp18:
	.loc	1 31 8                  // /tmp/dbg.cc:31:8
	ldur	w0, [x29, #-4]
.Ltmp19:
	//DEBUG_VALUE: func3:x <- $w0
	.loc	1 31 3 is_stmt 0        // /tmp/dbg.cc:31:3
	bl	usev
.Ltmp20:
	.loc	1 32 1 is_stmt 1        // /tmp/dbg.cc:32:1
	ldp	x29, x30, [sp, #16]     // 16-byte Folded Reload
	add	sp, sp, #32             // =32
	ret
.Ltmp21:
.Lfunc_end4:
	.size	func3, .Lfunc_end4-func3
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
	.asciz	"use"                   // string offset=136
.Linfo_string4:
	.asciz	"int"                   // string offset=140
.Linfo_string5:
	.asciz	"usev"                  // string offset=144
.Linfo_string6:
	.asciz	"useV"                  // string offset=149
.Linfo_string7:
	.asciz	"func00"                // string offset=154
.Linfo_string8:
	.asciz	"func0"                 // string offset=161
.Linfo_string9:
	.asciz	"func1"                 // string offset=167
.Linfo_string10:
	.asciz	"func2"                 // string offset=173
.Linfo_string11:
	.asciz	"func3"                 // string offset=179
.Linfo_string12:
	.asciz	"x"                     // string offset=185
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.xword	.Ltmp6-.Lfunc_begin0
	.xword	.Ltmp8-.Lfunc_begin0
	.hword	2                       // Loc expr size
	.byte	141                     // DW_OP_breg29
	.byte	124                     // -4
	.xword	.Ltmp8-.Lfunc_begin0
	.xword	.Ltmp9-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	80                      // DW_OP_reg0
	.xword	0
	.xword	0
.Ldebug_loc1:
	.xword	.Ltmp11-.Lfunc_begin0
	.xword	.Ltmp12-.Lfunc_begin0
	.hword	3                       // Loc expr size
	.byte	17                      // DW_OP_consts
	.byte	1                       // 1
	.byte	159                     // DW_OP_stack_value
	.xword	.Ltmp12-.Lfunc_begin0
	.xword	.Ltmp14-.Lfunc_begin0
	.hword	2                       // Loc expr size
	.byte	141                     // DW_OP_breg29
	.byte	124                     // -4
	.xword	.Ltmp14-.Lfunc_begin0
	.xword	.Ltmp15-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	80                      // DW_OP_reg0
	.xword	0
	.xword	0
.Ldebug_loc2:
	.xword	.Ltmp17-.Lfunc_begin0
	.xword	.Ltmp19-.Lfunc_begin0
	.hword	3                       // Loc expr size
	.byte	17                      // DW_OP_consts
	.byte	1                       // 1
	.byte	159                     // DW_OP_stack_value
	.xword	.Ltmp19-.Lfunc_begin0
	.xword	.Ltmp20-.Lfunc_begin0
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
	.byte	15                      // DW_TAG_pointer_type
	.byte	0                       // DW_CHILDREN_no
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	7                       // Abbreviation Code
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
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	63                      // DW_AT_external
	.byte	25                      // DW_FORM_flag_present
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	8                       // Abbreviation Code
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
	.byte	9                       // Abbreviation Code
	.ascii	"\211\202\001"          // DW_TAG_GNU_call_site
	.byte	0                       // DW_CHILDREN_no
	.byte	49                      // DW_AT_abstract_origin
	.byte	19                      // DW_FORM_ref4
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	10                      // Abbreviation Code
	.byte	52                      // DW_TAG_variable
	.byte	0                       // DW_CHILDREN_no
	.byte	2                       // DW_AT_location
	.byte	23                      // DW_FORM_sec_offset
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
	.byte	0                       // EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	.Ldebug_info_end0-.Ldebug_info_start0 // Length of Unit
.Ldebug_info_start0:
	.hword	4                       // DWARF version number
	.word	.debug_abbrev           // Offset Into Abbrev. Section
	.byte	8                       // Address Size (in bytes)
	.byte	1                       // Abbrev [1] 0xb:0x173 DW_TAG_compile_unit
	.word	.Linfo_string0          // DW_AT_producer
	.hword	33                      // DW_AT_language
	.word	.Linfo_string1          // DW_AT_name
	.word	.Lline_table_start0     // DW_AT_stmt_list
	.word	.Linfo_string2          // DW_AT_comp_dir
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end4-.Lfunc_begin0 // DW_AT_high_pc
	.byte	2                       // Abbrev [2] 0x2a:0xd DW_TAG_subprogram
	.word	.Linfo_string3          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	3                       // DW_AT_decl_line
                                        // DW_AT_declaration
                                        // DW_AT_external
	.byte	3                       // Abbrev [3] 0x31:0x5 DW_TAG_formal_parameter
	.word	55                      // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	4                       // Abbrev [4] 0x37:0x5 DW_TAG_pointer_type
	.word	60                      // DW_AT_type
	.byte	5                       // Abbrev [5] 0x3c:0x7 DW_TAG_base_type
	.word	.Linfo_string4          // DW_AT_name
	.byte	5                       // DW_AT_encoding
	.byte	4                       // DW_AT_byte_size
	.byte	2                       // Abbrev [2] 0x43:0xd DW_TAG_subprogram
	.word	.Linfo_string5          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	4                       // DW_AT_decl_line
                                        // DW_AT_declaration
                                        // DW_AT_external
	.byte	3                       // Abbrev [3] 0x4a:0x5 DW_TAG_formal_parameter
	.word	60                      // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	2                       // Abbrev [2] 0x50:0xd DW_TAG_subprogram
	.word	.Linfo_string6          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	2                       // DW_AT_decl_line
                                        // DW_AT_declaration
                                        // DW_AT_external
	.byte	3                       // Abbrev [3] 0x57:0x5 DW_TAG_formal_parameter
	.word	93                      // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	6                       // Abbrev [6] 0x5d:0x1 DW_TAG_pointer_type
	.byte	7                       // Abbrev [7] 0x5e:0x31 DW_TAG_subprogram
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_GNU_all_call_sites
	.word	.Linfo_string7          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	6                       // DW_AT_decl_line
                                        // DW_AT_external
	.byte	8                       // Abbrev [8] 0x73:0xe DW_TAG_variable
	.byte	2                       // DW_AT_location
	.byte	141                     // DW_OP_breg29
	.byte	124                     // -4
	.word	.Linfo_string12         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	7                       // DW_AT_decl_line
	.word	60                      // DW_AT_type
	.byte	9                       // Abbrev [9] 0x81:0xd DW_TAG_GNU_call_site
	.word	42                      // DW_AT_abstract_origin
	.xword	.Ltmp1                  // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	7                       // Abbrev [7] 0x8f:0x31 DW_TAG_subprogram
	.xword	.Lfunc_begin1           // DW_AT_low_pc
	.word	.Lfunc_end1-.Lfunc_begin1 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_GNU_all_call_sites
	.word	.Linfo_string8          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	11                      // DW_AT_decl_line
                                        // DW_AT_external
	.byte	8                       // Abbrev [8] 0xa4:0xe DW_TAG_variable
	.byte	2                       // DW_AT_location
	.byte	145
	.byte	124
	.word	.Linfo_string12         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	12                      // DW_AT_decl_line
	.word	60                      // DW_AT_type
	.byte	9                       // Abbrev [9] 0xb2:0xd DW_TAG_GNU_call_site
	.word	42                      // DW_AT_abstract_origin
	.xword	.Ltmp4                  // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	7                       // Abbrev [7] 0xc0:0x3f DW_TAG_subprogram
	.xword	.Lfunc_begin2           // DW_AT_low_pc
	.word	.Lfunc_end2-.Lfunc_begin2 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_GNU_all_call_sites
	.word	.Linfo_string9          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	16                      // DW_AT_decl_line
                                        // DW_AT_external
	.byte	10                      // Abbrev [10] 0xd5:0xf DW_TAG_variable
	.word	.Ldebug_loc0            // DW_AT_location
	.word	.Linfo_string12         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	17                      // DW_AT_decl_line
	.word	60                      // DW_AT_type
	.byte	9                       // Abbrev [9] 0xe4:0xd DW_TAG_GNU_call_site
	.word	42                      // DW_AT_abstract_origin
	.xword	.Ltmp7                  // DW_AT_low_pc
	.byte	9                       // Abbrev [9] 0xf1:0xd DW_TAG_GNU_call_site
	.word	67                      // DW_AT_abstract_origin
	.xword	.Ltmp9                  // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	7                       // Abbrev [7] 0xff:0x3f DW_TAG_subprogram
	.xword	.Lfunc_begin3           // DW_AT_low_pc
	.word	.Lfunc_end3-.Lfunc_begin3 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_GNU_all_call_sites
	.word	.Linfo_string10         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	22                      // DW_AT_decl_line
                                        // DW_AT_external
	.byte	10                      // Abbrev [10] 0x114:0xf DW_TAG_variable
	.word	.Ldebug_loc1            // DW_AT_location
	.word	.Linfo_string12         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	23                      // DW_AT_decl_line
	.word	60                      // DW_AT_type
	.byte	9                       // Abbrev [9] 0x123:0xd DW_TAG_GNU_call_site
	.word	42                      // DW_AT_abstract_origin
	.xword	.Ltmp13                 // DW_AT_low_pc
	.byte	9                       // Abbrev [9] 0x130:0xd DW_TAG_GNU_call_site
	.word	67                      // DW_AT_abstract_origin
	.xword	.Ltmp15                 // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	7                       // Abbrev [7] 0x13e:0x3f DW_TAG_subprogram
	.xword	.Lfunc_begin4           // DW_AT_low_pc
	.word	.Lfunc_end4-.Lfunc_begin4 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_GNU_all_call_sites
	.word	.Linfo_string11         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	28                      // DW_AT_decl_line
                                        // DW_AT_external
	.byte	10                      // Abbrev [10] 0x153:0xf DW_TAG_variable
	.word	.Ldebug_loc2            // DW_AT_location
	.word	.Linfo_string12         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	29                      // DW_AT_decl_line
	.word	60                      // DW_AT_type
	.byte	9                       // Abbrev [9] 0x162:0xd DW_TAG_GNU_call_site
	.word	80                      // DW_AT_abstract_origin
	.xword	.Ltmp18                 // DW_AT_low_pc
	.byte	9                       // Abbrev [9] 0x16f:0xd DW_TAG_GNU_call_site
	.word	67                      // DW_AT_abstract_origin
	.xword	.Ltmp20                 // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	0                       // End Of Children Mark
.Ldebug_info_end0:
	.ident	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 5d25e153457e7bdd4181ab7496402a565a5b9370)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
