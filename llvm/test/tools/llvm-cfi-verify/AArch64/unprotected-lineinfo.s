# RUN: llvm-mc %s -filetype obj -triple aarch64-- -o %t.o
# RUN: llvm-cfi-verify %t.o | FileCheck %s

# CHECK-LABEL: {{^Instruction: .* \(FAIL_BAD_CONDITIONAL_BRANCH\)}}
# CHECK-NEXT: tiny.cc:9

# CHECK: Expected Protected: 0 (0.00%)
# CHECK: Unexpected Protected: 0 (0.00%)
# CHECK: Expected Unprotected: 0 (0.00%)
# CHECK: Unexpected Unprotected (BAD): 1 (100.00%)

# Source (tiny.cc):
#   int a() { return 42; }
#   int b() { return 137; }
#   int main(int argc, char** argv) {
#     int(*ptr)();
#     if (argc == 1)
#       ptr = &a;
#     else
#       ptr = &b;
#     return ptr();
#   }
# Compile with:
#    clang++ -target aarch64-- -gmlt tiny.cc -S -o tiny.s
	.text
	.file	"tiny.cc"
	.globl	_Z1av
	.p2align	2
	.type	_Z1av,@function
_Z1av:                                  // @_Z1av
.Lfunc_begin0:
	.file	1 "tiny.cc"
	.loc	1 1 0                   // tiny.cc:1:0
	.cfi_startproc
// BB#0:
	mov	w0, #42
.Ltmp0:
	.loc	1 1 11 prologue_end     // tiny.cc:1:11
	ret
.Ltmp1:
.Lfunc_end0:
	.size	_Z1av, .Lfunc_end0-_Z1av
	.cfi_endproc

	.globl	_Z1bv
	.p2align	2
	.type	_Z1bv,@function
_Z1bv:                                  // @_Z1bv
.Lfunc_begin1:
	.loc	1 2 0                   // tiny.cc:2:0
	.cfi_startproc
// BB#0:
	mov	w0, #137
.Ltmp2:
	.loc	1 2 11 prologue_end     // tiny.cc:2:11
	ret
.Ltmp3:
.Lfunc_end1:
	.size	_Z1bv, .Lfunc_end1-_Z1bv
	.cfi_endproc

	.globl	main
	.p2align	2
	.type	main,@function
main:                                   // @main
.Lfunc_begin2:
	.loc	1 3 0                   // tiny.cc:3:0
	.cfi_startproc
// BB#0:
	sub	sp, sp, #48             // =48
	stp	x29, x30, [sp, #32]     // 8-byte Folded Spill
	add	x29, sp, #32            // =32
.Lcfi0:
	.cfi_def_cfa w29, 16
.Lcfi1:
	.cfi_offset w30, -8
.Lcfi2:
	.cfi_offset w29, -16
	stur	wzr, [x29, #-4]
	stur	w0, [x29, #-8]
	str	x1, [sp, #16]
.Ltmp4:
	.loc	1 5 7 prologue_end      // tiny.cc:5:7
	ldur	w0, [x29, #-8]
	cmp		w0, #1          // =1
	b.ne	.LBB2_2
// BB#1:
	.loc	1 0 7 is_stmt 0         // tiny.cc:0:7
	adrp	x8, _Z1av
	add	x8, x8, :lo12:_Z1av
	.loc	1 6 9 is_stmt 1         // tiny.cc:6:9
	str	x8, [sp, #8]
	.loc	1 6 5 is_stmt 0         // tiny.cc:6:5
	b	.LBB2_3
.LBB2_2:
	.loc	1 0 5                   // tiny.cc:0:5
	adrp	x8, _Z1bv
	add	x8, x8, :lo12:_Z1bv
	.loc	1 8 9 is_stmt 1         // tiny.cc:8:9
	str	x8, [sp, #8]
.LBB2_3:
	.loc	1 9 10                  // tiny.cc:9:10
	ldr	x8, [sp, #8]
	blr	x8
	.loc	1 9 3 is_stmt 0         // tiny.cc:9:3
	ldp	x29, x30, [sp, #32]     // 8-byte Folded Reload
	add	sp, sp, #48             // =48
	ret
.Ltmp5:
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc

	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 4.0.1-10 (tags/RELEASE_401/final)" // string offset=0
.Linfo_string1:
	.asciz	"tiny.cc"               // string offset=48
.Linfo_string2:
	.asciz	"/tmp"                  // string offset=56
	.section	.debug_loc,"",@progbits
	.section	.debug_abbrev,"",@progbits
.Lsection_abbrev:
	.byte	1                       // Abbreviation Code
	.byte	17                      // DW_TAG_compile_unit
	.byte	0                       // DW_CHILDREN_no
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
	.byte	0                       // EOM(3)
	.section	.debug_info,"",@progbits
.Lsection_info:
.Lcu_begin0:
	.word	38                      // Length of Unit
	.hword	4                       // DWARF version number
	.word	.Lsection_abbrev        // Offset Into Abbrev. Section
	.byte	8                       // Address Size (in bytes)
	.byte	1                       // Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.word	.Linfo_string0          // DW_AT_producer
	.hword	4                       // DW_AT_language
	.word	.Linfo_string1          // DW_AT_name
	.word	.Lline_table_start0     // DW_AT_stmt_list
	.word	.Linfo_string2          // DW_AT_comp_dir
	.xword	.Lfunc_begin0           // DW_AT_low_pc
	.word	.Lfunc_end2-.Lfunc_begin0 // DW_AT_high_pc
	.section	.debug_ranges,"",@progbits
.Ldebug_range:
	.section	.debug_macinfo,"",@progbits
.Ldebug_macinfo:
.Lcu_macro_begin0:
	.byte	0                       // End Of Macro List Mark

	.ident	"clang version 4.0.1-10 (tags/RELEASE_401/final)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
