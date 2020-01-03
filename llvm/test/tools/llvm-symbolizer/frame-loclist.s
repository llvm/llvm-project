// Test various single-location and location list formats.
// REQUIRES: aarch64-registered-target

// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android -o %t.o %s
// RUN: echo 'FRAME %t.o 0x4'  | llvm-symbolizer | FileCheck %s --check-prefix=CHECK0
// RUN: echo 'FRAME %t.o 0x24' | llvm-symbolizer | FileCheck %s --check-prefix=CHECK1
// RUN: echo 'FRAME %t.o 0x44' | llvm-symbolizer | FileCheck %s --check-prefix=CHECK2
// RUN: echo 'FRAME %t.o 0x68' | llvm-symbolizer | FileCheck %s --check-prefix=CHECK3
// RUN: echo 'FRAME %t.o 0x94' | llvm-symbolizer | FileCheck %s --check-prefix=CHECK4

// Built from the following source with
// clang -target aarch64-linux-android -O3 -g -S
// and edited to replace DW_OP_fbreg with DW_OP_breg29 in func00 (search for EDIT below).
//
// void use(void *);
// void usei(int);
// void func00() {
//   int x;
//   use(&x);
// }
//
// void func0() {
//   int x;
//   use(&x);
// }
//
// int func1() {
//   int x;
//   use(&x);
//   return x;
// }
//
// int func2() {
//   int x = 1;
//   use(&x);
//   return x;
// }
//
// int func3(int b) {
//   int x = b;
//   usei(x);
//   return x;
// }

// DW_AT_location        (DW_OP_breg29 W29-4)
// CHECK0:      func00
// CHECK0-NEXT: x
// CHECK0-NEXT: {{.*}}dbg.c:4
// CHECK0-NEXT: -4 4 ??

// DW_AT_location        (DW_OP_fbreg -4)
// CHECK1:      func0{{$}}
// CHECK1-NEXT: x
// CHECK1-NEXT: {{.*}}dbg.c:9
// CHECK1-NEXT: -4 4 ??

// DW_AT_location        (0x00000000: 
//   [0x000000000000004c, 0x0000000000000058): DW_OP_breg29 W29-4
//   [0x0000000000000058, 0x000000000000005c): DW_OP_reg0 W0)
// CHECK2:      func1
// CHECK2-NEXT: x
// CHECK2-NEXT: {{.*}}dbg.c:14
// CHECK2-NEXT: -4 4 ??

// DW_AT_location        (0x00000037: 
//    [0x0000000000000078, 0x0000000000000080): DW_OP_consts +1, DW_OP_stack_value
//    [0x0000000000000080, 0x0000000000000088): DW_OP_breg29 W29-4
//    [0x0000000000000088, 0x000000000000008c): DW_OP_reg0 W0)
// CHECK3:      func2
// CHECK3-NEXT: x
// CHECK3-NEXT: {{.*}}dbg.c:20
// CHECK3-NEXT: -4 4 ??

// No stack location.
// DW_AT_location        (0x00000083: 
//     [0x0000000000000090, 0x00000000000000a0): DW_OP_reg0 W0
//     [0x00000000000000a0, 0x00000000000000ac): DW_OP_reg19 W19)
// CHECK4:      func3
// CHECK4-NEXT: b
// CHECK4-NEXT: {{.*}}dbg.c:25
// CHECK4-NEXT: ?? 4 ??
// CHECK4-NEXT: func3
// CHECK4-NEXT: x
// CHECK4-NEXT: {{.*}}dbg.c:26
// CHECK4-NEXT: ?? 4 ??
//
	.text
	.file	"dbg.c"
	.globl	func00                  // -- Begin function func00
	.p2align	2
	.type	func00,@function
func00:                                 // @func00
.Lfunc_begin0:
	.file	1 "/tmp" "dbg.c"
	.loc	1 3 0                   // /tmp/dbg.c:3:0
	.cfi_sections .debug_frame
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
	.loc	1 5 3 prologue_end      // /tmp/dbg.c:5:3
	sub	x0, x29, #4             // =4
	bl	use
.Ltmp1:
	.loc	1 6 1                   // /tmp/dbg.c:6:1
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
	.loc	1 8 0                   // /tmp/dbg.c:8:0
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
	.loc	1 10 3 prologue_end     // /tmp/dbg.c:10:3
	sub	x0, x29, #4             // =4
	bl	use
.Ltmp4:
	.loc	1 11 1                  // /tmp/dbg.c:11:1
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
	.loc	1 13 0                  // /tmp/dbg.c:13:0
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
	.loc	1 15 3 prologue_end     // /tmp/dbg.c:15:3
	sub	x0, x29, #4             // =4
	bl	use
.Ltmp7:
	.loc	1 16 10                 // /tmp/dbg.c:16:10
	ldur	w0, [x29, #-4]
.Ltmp8:
	//DEBUG_VALUE: func1:x <- $w0
	.loc	1 16 3 is_stmt 0        // /tmp/dbg.c:16:3
	ldp	x29, x30, [sp, #16]     // 16-byte Folded Reload
	add	sp, sp, #32             // =32
	ret
.Ltmp9:
.Lfunc_end2:
	.size	func1, .Lfunc_end2-func1
	.cfi_endproc
                                        // -- End function
	.globl	func2                   // -- Begin function func2
	.p2align	2
	.type	func2,@function
func2:                                  // @func2
.Lfunc_begin3:
	.loc	1 19 0 is_stmt 1        // /tmp/dbg.c:19:0
	.cfi_startproc
// %bb.0:                               // %entry
	sub	sp, sp, #32             // =32
	stp	x29, x30, [sp, #16]     // 16-byte Folded Spill
	add	x29, sp, #16            // =16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	w8, #1
.Ltmp10:
	//DEBUG_VALUE: func2:x <- 1
	.loc	1 21 3 prologue_end     // /tmp/dbg.c:21:3
	sub	x0, x29, #4             // =4
	.loc	1 20 7                  // /tmp/dbg.c:20:7
	stur	w8, [x29, #-4]
.Ltmp11:
	//DEBUG_VALUE: func2:x <- [DW_OP_constu 4, DW_OP_minus, DW_OP_deref] $fp
	.loc	1 21 3                  // /tmp/dbg.c:21:3
	bl	use
.Ltmp12:
	.loc	1 22 10                 // /tmp/dbg.c:22:10
	ldur	w0, [x29, #-4]
.Ltmp13:
	//DEBUG_VALUE: func2:x <- $w0
	.loc	1 22 3 is_stmt 0        // /tmp/dbg.c:22:3
	ldp	x29, x30, [sp, #16]     // 16-byte Folded Reload
	add	sp, sp, #32             // =32
	ret
.Ltmp14:
.Lfunc_end3:
	.size	func2, .Lfunc_end3-func2
	.cfi_endproc
                                        // -- End function
	.globl	func3                   // -- Begin function func3
	.p2align	2
	.type	func3,@function
func3:                                  // @func3
.Lfunc_begin4:
	.loc	1 25 0 is_stmt 1        // /tmp/dbg.c:25:0
	.cfi_startproc
// %bb.0:                               // %entry
	//DEBUG_VALUE: func3:b <- $w0
	stp	x29, x30, [sp, #-32]!   // 16-byte Folded Spill
	str	x19, [sp, #16]          // 8-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
.Ltmp15:
	//DEBUG_VALUE: func3:x <- $w0
	mov	w19, w0
.Ltmp16:
	//DEBUG_VALUE: func3:x <- $w19
	//DEBUG_VALUE: func3:b <- $w19
	.loc	1 27 3 prologue_end     // /tmp/dbg.c:27:3
	bl	usei
.Ltmp17:
	.loc	1 28 3                  // /tmp/dbg.c:28:3
	mov	w0, w19
	ldr	x19, [sp, #16]          // 8-byte Folded Reload
.Ltmp18:
	ldp	x29, x30, [sp], #32     // 16-byte Folded Reload
	ret
.Ltmp19:
.Lfunc_end4:
	.size	func3, .Lfunc_end4-func3
	.cfi_endproc
                                        // -- End function
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 092ef9c6cf4678d2b8df7738110ecd405fe1fe3d)" // string offset=0
.Linfo_string1:
	.asciz	"/tmp/dbg.c"            // string offset=101
.Linfo_string2:
	.asciz	"/code/build-llvm-cmake" // string offset=112
.Linfo_string3:
	.asciz	"use"                   // string offset=135
.Linfo_string4:
	.asciz	"usei"                  // string offset=139
.Linfo_string5:
	.asciz	"int"                   // string offset=144
.Linfo_string6:
	.asciz	"func00"                // string offset=148
.Linfo_string7:
	.asciz	"func0"                 // string offset=155
.Linfo_string8:
	.asciz	"func1"                 // string offset=161
.Linfo_string9:
	.asciz	"func2"                 // string offset=167
.Linfo_string10:
	.asciz	"func3"                 // string offset=173
.Linfo_string11:
	.asciz	"x"                     // string offset=179
.Linfo_string12:
	.asciz	"b"                     // string offset=181
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.xword	.Ltmp6-.Lfunc_begin0
	.xword	.Ltmp8-.Lfunc_begin0
	.hword	2                       // Loc expr size
	.byte	141                     // DW_OP_breg29
	.byte	124                     // -4
	.xword	.Ltmp8-.Lfunc_begin0
	.xword	.Lfunc_end2-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	80                      // DW_OP_reg0
	.xword	0
	.xword	0
.Ldebug_loc1:
	.xword	.Ltmp10-.Lfunc_begin0
	.xword	.Ltmp11-.Lfunc_begin0
	.hword	3                       // Loc expr size
	.byte	17                      // DW_OP_consts
	.byte	1                       // 1
	.byte	159                     // DW_OP_stack_value
	.xword	.Ltmp11-.Lfunc_begin0
	.xword	.Ltmp13-.Lfunc_begin0
	.hword	2                       // Loc expr size
	.byte	141                     // DW_OP_breg29
	.byte	124                     // -4
	.xword	.Ltmp13-.Lfunc_begin0
	.xword	.Lfunc_end3-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	80                      // DW_OP_reg0
	.xword	0
	.xword	0
.Ldebug_loc2:
	.xword	.Lfunc_begin4-.Lfunc_begin0
	.xword	.Ltmp16-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	80                      // DW_OP_reg0
	.xword	.Ltmp16-.Lfunc_begin0
	.xword	.Ltmp18-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	99                      // DW_OP_reg19
	.xword	0
	.xword	0
.Ldebug_loc3:
	.xword	.Ltmp15-.Lfunc_begin0
	.xword	.Ltmp16-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	80                      // DW_OP_reg0
	.xword	.Ltmp16-.Lfunc_begin0
	.xword	.Ltmp18-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	99                      // DW_OP_reg19
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
	.byte	3                       // Abbreviation Code
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
	.byte	4                       // Abbreviation Code
	.ascii	"\211\202\001"          // DW_TAG_GNU_call_site
	.byte	0                       // DW_CHILDREN_no
	.byte	49                      // DW_AT_abstract_origin
	.byte	19                      // DW_FORM_ref4
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	5                       // Abbreviation Code
	.byte	46                      // DW_TAG_subprogram
	.byte	1                       // DW_CHILDREN_yes
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	39                      // DW_AT_prototyped
	.byte	25                      // DW_FORM_flag_present
	.byte	60                      // DW_AT_declaration
	.byte	25                      // DW_FORM_flag_present
	.byte	63                      // DW_AT_external
	.byte	25                      // DW_FORM_flag_present
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	6                       // Abbreviation Code
	.byte	5                       // DW_TAG_formal_parameter
	.byte	0                       // DW_CHILDREN_no
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	7                       // Abbreviation Code
	.byte	15                      // DW_TAG_pointer_type
	.byte	0                       // DW_CHILDREN_no
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
	.ascii	"\227B"                 // DW_AT_GNU_all_call_sites
	.byte	25                      // DW_FORM_flag_present
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	63                      // DW_AT_external
	.byte	25                      // DW_FORM_flag_present
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	9                       // Abbreviation Code
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
	.byte	10                      // Abbreviation Code
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
	.byte	39                      // DW_AT_prototyped
	.byte	25                      // DW_FORM_flag_present
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	63                      // DW_AT_external
	.byte	25                      // DW_FORM_flag_present
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	11                      // Abbreviation Code
	.byte	5                       // DW_TAG_formal_parameter
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
	.byte	12                      // Abbreviation Code
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
	.byte	1                       // Abbrev [1] 0xb:0x155 DW_TAG_compile_unit
	.word	.Linfo_string0          // DW_AT_producer
	.hword	12                      // DW_AT_language
	.word	.Linfo_string1          // DW_AT_name
	.word	.Lline_table_start0     // DW_AT_stmt_list
	.word	.Linfo_string2          // DW_AT_comp_dir
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end4-.Lfunc_begin0 // DW_AT_high_pc
	.byte	2                       // Abbrev [2] 0x2a:0x31 DW_TAG_subprogram
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_GNU_all_call_sites
	.word	.Linfo_string6          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	3                       // DW_AT_decl_line
                                        // DW_AT_external
	.byte	3                       // Abbrev [3] 0x3f:0xe DW_TAG_variable
	.byte	2                       // DW_AT_location
	.byte	141                     // DW_OP_breg29  !!! EDIT: 145 (fbreg) to 141 (breg29)
	.byte	124                     // -4
	.word	.Linfo_string11         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	4                       // DW_AT_decl_line
	.word	344                     // DW_AT_type
	.byte	4                       // Abbrev [4] 0x4d:0xd DW_TAG_GNU_call_site
	.word	91                      // DW_AT_abstract_origin
	.xword	.Ltmp1                  // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	5                       // Abbrev [5] 0x5b:0xd DW_TAG_subprogram
	.word	.Linfo_string3          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	1                       // DW_AT_decl_line
                                        // DW_AT_prototyped
                                        // DW_AT_declaration
                                        // DW_AT_external
	.byte	6                       // Abbrev [6] 0x62:0x5 DW_TAG_formal_parameter
	.word	104                     // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	7                       // Abbrev [7] 0x68:0x1 DW_TAG_pointer_type
	.byte	2                       // Abbrev [2] 0x69:0x31 DW_TAG_subprogram
	.xword	.Lfunc_begin1           // DW_AT_low_pc
	.word	.Lfunc_end1-.Lfunc_begin1 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_GNU_all_call_sites
	.word	.Linfo_string7          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	8                       // DW_AT_decl_line
                                        // DW_AT_external
	.byte	3                       // Abbrev [3] 0x7e:0xe DW_TAG_variable
	.byte	2                       // DW_AT_location
	.byte	145
	.byte	124
	.word	.Linfo_string11         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	9                       // DW_AT_decl_line
	.word	344                     // DW_AT_type
	.byte	4                       // Abbrev [4] 0x8c:0xd DW_TAG_GNU_call_site
	.word	91                      // DW_AT_abstract_origin
	.xword	.Ltmp4                  // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	8                       // Abbrev [8] 0x9a:0x36 DW_TAG_subprogram
	.xword	.Lfunc_begin2           // DW_AT_low_pc
	.word	.Lfunc_end2-.Lfunc_begin2 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_GNU_all_call_sites
	.word	.Linfo_string8          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	13                      // DW_AT_decl_line
	.word	344                     // DW_AT_type
                                        // DW_AT_external
	.byte	9                       // Abbrev [9] 0xb3:0xf DW_TAG_variable
	.word	.Ldebug_loc0            // DW_AT_location
	.word	.Linfo_string11         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	14                      // DW_AT_decl_line
	.word	344                     // DW_AT_type
	.byte	4                       // Abbrev [4] 0xc2:0xd DW_TAG_GNU_call_site
	.word	91                      // DW_AT_abstract_origin
	.xword	.Ltmp7                  // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	8                       // Abbrev [8] 0xd0:0x36 DW_TAG_subprogram
	.xword	.Lfunc_begin3           // DW_AT_low_pc
	.word	.Lfunc_end3-.Lfunc_begin3 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_GNU_all_call_sites
	.word	.Linfo_string9          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	19                      // DW_AT_decl_line
	.word	344                     // DW_AT_type
                                        // DW_AT_external
	.byte	9                       // Abbrev [9] 0xe9:0xf DW_TAG_variable
	.word	.Ldebug_loc1            // DW_AT_location
	.word	.Linfo_string11         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	20                      // DW_AT_decl_line
	.word	344                     // DW_AT_type
	.byte	4                       // Abbrev [4] 0xf8:0xd DW_TAG_GNU_call_site
	.word	91                      // DW_AT_abstract_origin
	.xword	.Ltmp12                 // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	10                      // Abbrev [10] 0x106:0x45 DW_TAG_subprogram
	.xword	.Lfunc_begin4           // DW_AT_low_pc
	.word	.Lfunc_end4-.Lfunc_begin4 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
                                        // DW_AT_GNU_all_call_sites
	.word	.Linfo_string10         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	25                      // DW_AT_decl_line
                                        // DW_AT_prototyped
	.word	344                     // DW_AT_type
                                        // DW_AT_external
	.byte	11                      // Abbrev [11] 0x11f:0xf DW_TAG_formal_parameter
	.word	.Ldebug_loc2            // DW_AT_location
	.word	.Linfo_string12         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	25                      // DW_AT_decl_line
	.word	344                     // DW_AT_type
	.byte	9                       // Abbrev [9] 0x12e:0xf DW_TAG_variable
	.word	.Ldebug_loc3            // DW_AT_location
	.word	.Linfo_string11         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	26                      // DW_AT_decl_line
	.word	344                     // DW_AT_type
	.byte	4                       // Abbrev [4] 0x13d:0xd DW_TAG_GNU_call_site
	.word	331                     // DW_AT_abstract_origin
	.xword	.Ltmp17                 // DW_AT_low_pc
	.byte	0                       // End Of Children Mark
	.byte	5                       // Abbrev [5] 0x14b:0xd DW_TAG_subprogram
	.word	.Linfo_string4          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	2                       // DW_AT_decl_line
                                        // DW_AT_prototyped
                                        // DW_AT_declaration
                                        // DW_AT_external
	.byte	6                       // Abbrev [6] 0x152:0x5 DW_TAG_formal_parameter
	.word	344                     // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	12                      // Abbrev [12] 0x158:0x7 DW_TAG_base_type
	.word	.Linfo_string5          // DW_AT_name
	.byte	5                       // DW_AT_encoding
	.byte	4                       // DW_AT_byte_size
	.byte	0                       // End Of Children Mark
.Ldebug_info_end0:
	.ident	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 092ef9c6cf4678d2b8df7738110ecd405fe1fe3d)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
