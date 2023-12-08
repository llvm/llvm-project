// REQUIRES: aarch64-registered-target

// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android -o %t.o %s
// RUN: echo 'FRAME %t.o 0' | llvm-symbolizer | FileCheck %s

// CHECK: f
// CHECK-NEXT: a
// CHECK-NEXT: /tmp{{/|\\}}stack.c:20
// CHECK-NEXT: -192 32 192
// CHECK-NEXT: g
// CHECK-NEXT: p
// CHECK-NEXT: /tmp{{/|\\}}stack.c:8
// CHECK-NEXT: ?? 8 ??
// CHECK-NEXT: g
// CHECK-NEXT: b
// CHECK-NEXT: /tmp{{/|\\}}stack.c:10
// CHECK-NEXT: -128 32 128
// CHECK-NEXT: h
// CHECK-NEXT: p1
// CHECK-NEXT: /tmp{{/|\\}}stack.c:3
// CHECK-NEXT: ?? 8 ??
// CHECK-NEXT: h
// CHECK-NEXT: p2
// CHECK-NEXT: /tmp{{/|\\}}stack.c:3
// CHECK-NEXT: ?? 8 ??
// CHECK-NEXT: h
// CHECK-NEXT: d
// CHECK-NEXT: /tmp{{/|\\}}stack.c:4
// CHECK-NEXT: -96 32 0
// CHECK-NEXT: g
// CHECK-NEXT: c
// CHECK-NEXT: /tmp{{/|\\}}stack.c:14
// CHECK-NEXT: -160 32 64
// CHECK-NEXT: h
// CHECK-NEXT: p1
// CHECK-NEXT: /tmp{{/|\\}}stack.c:3
// CHECK-NEXT: ?? 8 ??
// CHECK-NEXT: h
// CHECK-NEXT: p2
// CHECK-NEXT: /tmp{{/|\\}}stack.c:3
// CHECK-NEXT: ?? 8 ??
// CHECK-NEXT: h
// CHECK-NEXT: d
// CHECK-NEXT: /tmp{{/|\\}}stack.c:4
// CHECK-NEXT: -96 32 0

// Generated from:
//
// void i(void *, void *, void *);
//
// static void h(void *p1, void *p2) {
//   char d[32];
//   i(d, p1, p2);
// }
//
// static void g(void *p) {
//   {
//     char b[32];
//     h(b, p);
//   }
//   {
//     char c[32];
//     h(c, p);
//   }
// }
//
// clang -S -o - -fsanitize=hwaddress --target=aarch64-linux-android /tmp/stack.c -O -fsanitize-hwaddress-abi=platform -g

	.text
	.file	"stack.c"
	.globl	f                       // -- Begin function f
	.p2align	2
	.type	f,@function
f:                                      // @f
.Lfunc_begin0:
	.file	1 "/tmp" "stack.c"
	.loc	1 19 0                  // stack.c:19:0
	.cfi_startproc
// %bb.0:                               // %entry
	sub	sp, sp, #208            // =208
	stp	x26, x25, [sp, #128]    // 16-byte Folded Spill
	stp	x24, x23, [sp, #144]    // 16-byte Folded Spill
	stp	x22, x21, [sp, #160]    // 16-byte Folded Spill
	stp	x20, x19, [sp, #176]    // 16-byte Folded Spill
	stp	x29, x30, [sp, #192]    // 16-byte Folded Spill
	add	x29, sp, #192           // =192
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	mrs	x8, TPIDR_EL0
	ldr	x12, [x8, #48]
.Ltmp0:
	adr	x14, .Ltmp0
	orr	x14, x14, x29, lsl #44
	add	x9, sp, #96             // =96
	asr	x15, x12, #3
	asr	x16, x12, #56
	orr	x17, x12, #0xffffffff
	str	x14, [x12], #8
	bic	x12, x12, x16, lsl #12
	str	x12, [x8, #48]
.Ltmp1:
	.loc	1 4 8 prologue_end      // stack.c:4:8
	and	w8, w15, #0xff
	lsr	x22, x9, #4
	add	x25, x17, #1            // =1
	bfi	w8, w8, #8, #8
	add	x10, sp, #64            // =64
	strh	w8, [x25, x22]
.Ltmp2:
	.loc	1 10 10                 // stack.c:10:10
	eor	x8, x15, #0x80
	orr	x1, x10, x8, lsl #56
	and	w8, w8, #0xff
	lsr	x23, x10, #4
	bfi	w8, w8, #8, #8
	add	x11, sp, #32            // =32
	strh	w8, [x25, x23]
.Ltmp3:
	.loc	1 14 10                 // stack.c:14:10
	eor	x8, x15, #0x40
	orr	x19, x11, x8, lsl #56
	and	w8, w8, #0xff
	lsr	x24, x11, #4
	bfi	w8, w8, #8, #8
	mov	x13, sp
	strh	w8, [x25, x24]
.Ltmp4:
	.loc	1 20 3                  // stack.c:20:3
	eor	x8, x15, #0xc0
.Ltmp5:
	.loc	1 4 8                   // stack.c:4:8
	orr	x20, x9, x15, lsl #56
.Ltmp6:
	.loc	1 20 3                  // stack.c:20:3
	orr	x21, x13, x8, lsl #56
	and	w8, w8, #0xff
	lsr	x26, x13, #4
	bfi	w8, w8, #8, #8
.Ltmp7:
	.loc	1 5 3                   // stack.c:5:3
	mov	x0, x20
	mov	x2, x21
.Ltmp8:
	.loc	1 20 3                  // stack.c:20:3
	strh	w8, [x25, x26]
.Ltmp9:
	//DEBUG_VALUE: h:p1 <- $x1
	//DEBUG_VALUE: g:p <- $x21
	//DEBUG_VALUE: h:p2 <- $x21
	//DEBUG_VALUE: h:p2 <- $x21
	.loc	1 5 3                   // stack.c:5:3
	bl	i
.Ltmp10:
	//DEBUG_VALUE: h:p1 <- $x19
	.loc	1 5 3 is_stmt 0         // stack.c:5:3
	mov	x0, x20
	mov	x1, x19
	mov	x2, x21
	bl	i
.Ltmp11:
	.loc	1 22 1 is_stmt 1        // stack.c:22:1
	strh	wzr, [x25, x22]
	strh	wzr, [x25, x23]
	strh	wzr, [x25, x24]
	strh	wzr, [x25, x26]
	ldp	x29, x30, [sp, #192]    // 16-byte Folded Reload
	ldp	x20, x19, [sp, #176]    // 16-byte Folded Reload
.Ltmp12:
	ldp	x22, x21, [sp, #160]    // 16-byte Folded Reload
.Ltmp13:
	ldp	x24, x23, [sp, #144]    // 16-byte Folded Reload
	ldp	x26, x25, [sp, #128]    // 16-byte Folded Reload
	add	sp, sp, #208            // =208
	ret
.Ltmp14:
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
	.cfi_endproc
                                        // -- End function
	.section	.text.hwasan.module_ctor,"axG",@progbits,hwasan.module_ctor,comdat
	.p2align	2               // -- Begin function hwasan.module_ctor
	.type	hwasan.module_ctor,@function
hwasan.module_ctor:                     // @hwasan.module_ctor
.Lfunc_begin1:
	.cfi_startproc
// %bb.0:
	str	x30, [sp, #-16]!        // 8-byte Folded Spill
	.cfi_def_cfa_offset 16
	.cfi_offset w30, -16
	bl	__hwasan_init
	ldr	x30, [sp], #16          // 8-byte Folded Reload
	ret
.Lfunc_end1:
	.size	hwasan.module_ctor, .Lfunc_end1-hwasan.module_ctor
	.cfi_endproc
                                        // -- End function
	.section	.init_array.0,"aGw",@init_array,hwasan.module_ctor,comdat
	.p2align	3
	.xword	hwasan.module_ctor
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 9.0.0 "  // string offset=0
.Linfo_string1:
	.asciz	"stack.c"               // string offset=21
.Linfo_string2:
	.asciz	"/tmp"                  // string offset=29
.Linfo_string3:
	.asciz	"h"                     // string offset=34
.Linfo_string4:
	.asciz	"p1"                    // string offset=36
.Linfo_string5:
	.asciz	"p2"                    // string offset=39
.Linfo_string6:
	.asciz	"d"                     // string offset=42
.Linfo_string7:
	.asciz	"char"                  // string offset=44
.Linfo_string8:
	.asciz	"__ARRAY_SIZE_TYPE__"   // string offset=49
.Linfo_string9:
	.asciz	"g"                     // string offset=69
.Linfo_string10:
	.asciz	"p"                     // string offset=71
.Linfo_string11:
	.asciz	"b"                     // string offset=73
.Linfo_string12:
	.asciz	"c"                     // string offset=75
.Linfo_string13:
	.asciz	"f"                     // string offset=77
.Linfo_string14:
	.asciz	"a"                     // string offset=79
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.xword	.Ltmp9-.Lfunc_begin0
	.xword	.Ltmp10-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	81                      // DW_OP_reg1
	.xword	0
	.xword	0
.Ldebug_loc1:
	.xword	.Ltmp9-.Lfunc_begin0
	.xword	.Ltmp13-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	101                     // DW_OP_reg21
	.xword	0
	.xword	0
.Ldebug_loc2:
	.xword	.Ltmp9-.Lfunc_begin0
	.xword	.Ltmp13-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	101                     // DW_OP_reg21
	.xword	0
	.xword	0
.Ldebug_loc3:
	.xword	.Ltmp9-.Lfunc_begin0
	.xword	.Ltmp13-.Lfunc_begin0
	.hword	1                       // Loc expr size
	.byte	101                     // DW_OP_reg21
	.xword	0
	.xword	0
.Ldebug_loc4:
	.xword	.Ltmp10-.Lfunc_begin0
	.xword	.Ltmp12-.Lfunc_begin0
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
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	58                      // DW_AT_decl_file
	.byte	11                      // DW_FORM_data1
	.byte	59                      // DW_AT_decl_line
	.byte	11                      // DW_FORM_data1
	.byte	39                      // DW_AT_prototyped
	.byte	25                      // DW_FORM_flag_present
	.byte	32                      // DW_AT_inline
	.byte	11                      // DW_FORM_data1
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	3                       // Abbreviation Code
	.byte	5                       // DW_TAG_formal_parameter
	.byte	0                       // DW_CHILDREN_no
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
	.byte	52                      // DW_TAG_variable
	.byte	0                       // DW_CHILDREN_no
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
	.byte	5                       // Abbreviation Code
	.byte	15                      // DW_TAG_pointer_type
	.byte	0                       // DW_CHILDREN_no
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	6                       // Abbreviation Code
	.byte	1                       // DW_TAG_array_type
	.byte	1                       // DW_CHILDREN_yes
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	7                       // Abbreviation Code
	.byte	33                      // DW_TAG_subrange_type
	.byte	0                       // DW_CHILDREN_no
	.byte	73                      // DW_AT_type
	.byte	19                      // DW_FORM_ref4
	.byte	55                      // DW_AT_count
	.byte	11                      // DW_FORM_data1
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	8                       // Abbreviation Code
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
	.byte	9                       // Abbreviation Code
	.byte	36                      // DW_TAG_base_type
	.byte	0                       // DW_CHILDREN_no
	.byte	3                       // DW_AT_name
	.byte	14                      // DW_FORM_strp
	.byte	11                      // DW_AT_byte_size
	.byte	11                      // DW_FORM_data1
	.byte	62                      // DW_AT_encoding
	.byte	11                      // DW_FORM_data1
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	10                      // Abbreviation Code
	.byte	11                      // DW_TAG_lexical_block
	.byte	1                       // DW_CHILDREN_yes
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	11                      // Abbreviation Code
	.byte	46                      // DW_TAG_subprogram
	.byte	1                       // DW_CHILDREN_yes
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
	.byte	18                      // DW_AT_high_pc
	.byte	6                       // DW_FORM_data4
	.byte	64                      // DW_AT_frame_base
	.byte	24                      // DW_FORM_exprloc
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
	.byte	12                      // Abbreviation Code
	.byte	52                      // DW_TAG_variable
	.byte	0                       // DW_CHILDREN_no
	.byte	2                       // DW_AT_location
	.byte	24                      // DW_FORM_exprloc
	.ascii	"\203|"                 // DW_AT_LLVM_tag_offset
	.byte	11                      // DW_FORM_data1
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
	.byte	13                      // Abbreviation Code
	.byte	29                      // DW_TAG_inlined_subroutine
	.byte	1                       // DW_CHILDREN_yes
	.byte	49                      // DW_AT_abstract_origin
	.byte	19                      // DW_FORM_ref4
	.byte	85                      // DW_AT_ranges
	.byte	23                      // DW_FORM_sec_offset
	.byte	88                      // DW_AT_call_file
	.byte	11                      // DW_FORM_data1
	.byte	89                      // DW_AT_call_line
	.byte	11                      // DW_FORM_data1
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	14                      // Abbreviation Code
	.byte	5                       // DW_TAG_formal_parameter
	.byte	0                       // DW_CHILDREN_no
	.byte	2                       // DW_AT_location
	.byte	23                      // DW_FORM_sec_offset
	.byte	49                      // DW_AT_abstract_origin
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	15                      // Abbreviation Code
	.byte	11                      // DW_TAG_lexical_block
	.byte	1                       // DW_CHILDREN_yes
	.byte	85                      // DW_AT_ranges
	.byte	23                      // DW_FORM_sec_offset
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	16                      // Abbreviation Code
	.byte	52                      // DW_TAG_variable
	.byte	0                       // DW_CHILDREN_no
	.byte	2                       // DW_AT_location
	.byte	24                      // DW_FORM_exprloc
	.ascii	"\203|"                 // DW_AT_LLVM_tag_offset
	.byte	11                      // DW_FORM_data1
	.byte	49                      // DW_AT_abstract_origin
	.byte	19                      // DW_FORM_ref4
	.byte	0                       // EOM(1)
	.byte	0                       // EOM(2)
	.byte	17                      // Abbreviation Code
	.byte	29                      // DW_TAG_inlined_subroutine
	.byte	1                       // DW_CHILDREN_yes
	.byte	49                      // DW_AT_abstract_origin
	.byte	19                      // DW_FORM_ref4
	.byte	17                      // DW_AT_low_pc
	.byte	1                       // DW_FORM_addr
	.byte	18                      // DW_AT_high_pc
	.byte	6                       // DW_FORM_data4
	.byte	88                      // DW_AT_call_file
	.byte	11                      // DW_FORM_data1
	.byte	89                      // DW_AT_call_line
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
	.byte	1                       // Abbrev [1] 0xb:0x146 DW_TAG_compile_unit
	.word	.Linfo_string0          // DW_AT_producer
	.hword	12                      // DW_AT_language
	.word	.Linfo_string1          // DW_AT_name
	.word	.Lline_table_start0     // DW_AT_stmt_list
	.word	.Linfo_string2          // DW_AT_comp_dir
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.byte	2                       // Abbrev [2] 0x2a:0x2a DW_TAG_subprogram
	.word	.Linfo_string3          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	3                       // DW_AT_decl_line
                                        // DW_AT_prototyped
	.byte	1                       // DW_AT_inline
	.byte	3                       // Abbrev [3] 0x32:0xb DW_TAG_formal_parameter
	.word	.Linfo_string4          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	3                       // DW_AT_decl_line
	.word	84                      // DW_AT_type
	.byte	3                       // Abbrev [3] 0x3d:0xb DW_TAG_formal_parameter
	.word	.Linfo_string5          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	3                       // DW_AT_decl_line
	.word	84                      // DW_AT_type
	.byte	4                       // Abbrev [4] 0x48:0xb DW_TAG_variable
	.word	.Linfo_string6          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	4                       // DW_AT_decl_line
	.word	85                      // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	5                       // Abbrev [5] 0x54:0x1 DW_TAG_pointer_type
	.byte	6                       // Abbrev [6] 0x55:0xc DW_TAG_array_type
	.word	97                      // DW_AT_type
	.byte	7                       // Abbrev [7] 0x5a:0x6 DW_TAG_subrange_type
	.word	104                     // DW_AT_type
	.byte	32                      // DW_AT_count
	.byte	0                       // End Of Children Mark
	.byte	8                       // Abbrev [8] 0x61:0x7 DW_TAG_base_type
	.word	.Linfo_string7          // DW_AT_name
	.byte	8                       // DW_AT_encoding
	.byte	1                       // DW_AT_byte_size
	.byte	9                       // Abbrev [9] 0x68:0x7 DW_TAG_base_type
	.word	.Linfo_string8          // DW_AT_name
	.byte	8                       // DW_AT_byte_size
	.byte	7                       // DW_AT_encoding
	.byte	2                       // Abbrev [2] 0x6f:0x2e DW_TAG_subprogram
	.word	.Linfo_string9          // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	8                       // DW_AT_decl_line
                                        // DW_AT_prototyped
	.byte	1                       // DW_AT_inline
	.byte	3                       // Abbrev [3] 0x77:0xb DW_TAG_formal_parameter
	.word	.Linfo_string10         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	8                       // DW_AT_decl_line
	.word	84                      // DW_AT_type
	.byte	10                      // Abbrev [10] 0x82:0xd DW_TAG_lexical_block
	.byte	4                       // Abbrev [4] 0x83:0xb DW_TAG_variable
	.word	.Linfo_string11         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	10                      // DW_AT_decl_line
	.word	85                      // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	10                      // Abbrev [10] 0x8f:0xd DW_TAG_lexical_block
	.byte	4                       // Abbrev [4] 0x90:0xb DW_TAG_variable
	.word	.Linfo_string12         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	14                      // DW_AT_decl_line
	.word	85                      // DW_AT_type
	.byte	0                       // End Of Children Mark
	.byte	0                       // End Of Children Mark
	.byte	11                      // Abbrev [11] 0x9d:0xb3 DW_TAG_subprogram
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0 // DW_AT_high_pc
	.byte	1                       // DW_AT_frame_base
	.byte	109
	.word	.Linfo_string13         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	19                      // DW_AT_decl_line
                                        // DW_AT_external
	.byte	12                      // Abbrev [12] 0xb2:0x10 DW_TAG_variable
	.byte	3                       // DW_AT_location
	.byte	145
	.ascii	"\300~"
	.byte	192                     // DW_AT_LLVM_tag_offset
	.word	.Linfo_string14         // DW_AT_name
	.byte	1                       // DW_AT_decl_file
	.byte	20                      // DW_AT_decl_line
	.word	85                      // DW_AT_type
	.byte	13                      // Abbrev [13] 0xc2:0x8d DW_TAG_inlined_subroutine
	.word	111                     // DW_AT_abstract_origin
	.word	.Ldebug_ranges0         // DW_AT_ranges
	.byte	1                       // DW_AT_call_file
	.byte	21                      // DW_AT_call_line
	.byte	14                      // Abbrev [14] 0xcd:0x9 DW_TAG_formal_parameter
	.word	.Ldebug_loc1            // DW_AT_location
	.word	119                     // DW_AT_abstract_origin
	.byte	15                      // Abbrev [15] 0xd6:0x38 DW_TAG_lexical_block
	.word	.Ldebug_ranges2         // DW_AT_ranges
	.byte	16                      // Abbrev [16] 0xdb:0xa DW_TAG_variable
	.byte	3                       // DW_AT_location
	.byte	145
	.ascii	"\200\177"
	.byte	128                     // DW_AT_LLVM_tag_offset
	.word	131                     // DW_AT_abstract_origin
	.byte	13                      // Abbrev [13] 0xe5:0x28 DW_TAG_inlined_subroutine
	.word	42                      // DW_AT_abstract_origin
	.word	.Ldebug_ranges1         // DW_AT_ranges
	.byte	1                       // DW_AT_call_file
	.byte	11                      // DW_AT_call_line
	.byte	14                      // Abbrev [14] 0xf0:0x9 DW_TAG_formal_parameter
	.word	.Ldebug_loc0            // DW_AT_location
	.word	50                      // DW_AT_abstract_origin
	.byte	14                      // Abbrev [14] 0xf9:0x9 DW_TAG_formal_parameter
	.word	.Ldebug_loc2            // DW_AT_location
	.word	61                      // DW_AT_abstract_origin
	.byte	16                      // Abbrev [16] 0x102:0xa DW_TAG_variable
	.byte	3                       // DW_AT_location
	.byte	145
	.ascii	"\240\177"
	.byte	0                       // DW_AT_LLVM_tag_offset
	.word	72                      // DW_AT_abstract_origin
	.byte	0                       // End Of Children Mark
	.byte	0                       // End Of Children Mark
	.byte	15                      // Abbrev [15] 0x10e:0x40 DW_TAG_lexical_block
	.word	.Ldebug_ranges3         // DW_AT_ranges
	.byte	16                      // Abbrev [16] 0x113:0xa DW_TAG_variable
	.byte	3                       // DW_AT_location
	.byte	145
	.ascii	"\340~"
	.byte	64                      // DW_AT_LLVM_tag_offset
	.word	144                     // DW_AT_abstract_origin
	.byte	17                      // Abbrev [17] 0x11d:0x30 DW_TAG_inlined_subroutine
	.word	42                      // DW_AT_abstract_origin
	.xword	.Ltmp10                 // DW_AT_low_pc
	.word	.Ltmp11-.Ltmp10         // DW_AT_high_pc
	.byte	1                       // DW_AT_call_file
	.byte	15                      // DW_AT_call_line
	.byte	14                      // Abbrev [14] 0x130:0x9 DW_TAG_formal_parameter
	.word	.Ldebug_loc4            // DW_AT_location
	.word	50                      // DW_AT_abstract_origin
	.byte	14                      // Abbrev [14] 0x139:0x9 DW_TAG_formal_parameter
	.word	.Ldebug_loc3            // DW_AT_location
	.word	61                      // DW_AT_abstract_origin
	.byte	16                      // Abbrev [16] 0x142:0xa DW_TAG_variable
	.byte	3                       // DW_AT_location
	.byte	145
	.ascii	"\240\177"
	.byte	0                       // DW_AT_LLVM_tag_offset
	.word	72                      // DW_AT_abstract_origin
	.byte	0                       // End Of Children Mark
	.byte	0                       // End Of Children Mark
	.byte	0                       // End Of Children Mark
	.byte	0                       // End Of Children Mark
	.byte	0                       // End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.xword	.Ltmp1-.Lfunc_begin0
	.xword	.Ltmp4-.Lfunc_begin0
	.xword	.Ltmp5-.Lfunc_begin0
	.xword	.Ltmp6-.Lfunc_begin0
	.xword	.Ltmp7-.Lfunc_begin0
	.xword	.Ltmp8-.Lfunc_begin0
	.xword	.Ltmp9-.Lfunc_begin0
	.xword	.Ltmp11-.Lfunc_begin0
	.xword	0
	.xword	0
.Ldebug_ranges1:
	.xword	.Ltmp1-.Lfunc_begin0
	.xword	.Ltmp2-.Lfunc_begin0
	.xword	.Ltmp5-.Lfunc_begin0
	.xword	.Ltmp6-.Lfunc_begin0
	.xword	.Ltmp7-.Lfunc_begin0
	.xword	.Ltmp8-.Lfunc_begin0
	.xword	.Ltmp9-.Lfunc_begin0
	.xword	.Ltmp10-.Lfunc_begin0
	.xword	0
	.xword	0
.Ldebug_ranges2:
	.xword	.Ltmp1-.Lfunc_begin0
	.xword	.Ltmp3-.Lfunc_begin0
	.xword	.Ltmp5-.Lfunc_begin0
	.xword	.Ltmp6-.Lfunc_begin0
	.xword	.Ltmp7-.Lfunc_begin0
	.xword	.Ltmp8-.Lfunc_begin0
	.xword	.Ltmp9-.Lfunc_begin0
	.xword	.Ltmp10-.Lfunc_begin0
	.xword	0
	.xword	0
.Ldebug_ranges3:
	.xword	.Ltmp3-.Lfunc_begin0
	.xword	.Ltmp4-.Lfunc_begin0
	.xword	.Ltmp10-.Lfunc_begin0
	.xword	.Ltmp11-.Lfunc_begin0
	.xword	0
	.xword	0
	.section	.debug_macinfo,"",@progbits
	.byte	0                       // End Of Macro List Mark

	.ident	"clang version 9.0.0 "
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
