# RUN: llvm-mc %s -filetype obj -triple aarch64-- -o %t.o
# RUN: llvm-cfi-verify %t.o | FileCheck %s

# CHECK-LABEL: {{^Instruction: .* \(PROTECTED\)}}
# CHECK-NEXT: tiny.cc:9

# CHECK: Expected Protected: 1 (100.00%)
# CHECK: Unexpected Protected: 0 (0.00%)
# CHECK: Expected Unprotected: 0 (0.00%)
# CHECK: Unexpected Unprotected (BAD): 0 (0.00%)

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
	.text
	.file	"ld-temp.o"
	.p2align	2
	.type	_Z1av.cfi,@function
_Z1av.cfi:
.Lfunc_begin0:
	.file	1 "/tmp/tiny.cc"
	.loc	1 1 0
	.cfi_startproc
	.loc	1 1 11 prologue_end
	mov	w0, #42
	ret
.Ltmp0:
.Lfunc_end0:
	.size	_Z1av.cfi, .Lfunc_end0-_Z1av.cfi
	.cfi_endproc

	.p2align	2
	.type	_Z1bv.cfi,@function
_Z1bv.cfi:
.Lfunc_begin1:
	.loc	1 2 0
	.cfi_startproc
	.loc	1 2 11 prologue_end
	mov	w0, #137
	ret
.Ltmp1:
.Lfunc_end1:
	.size	_Z1bv.cfi, .Lfunc_end1-_Z1bv.cfi
	.cfi_endproc

	.p2align	2
	.type	main,@function
main:
.Lfunc_begin2:
	.loc	1 3 0
	.cfi_startproc
	sub	sp, sp, #48
	stp	x29, x30, [sp, #32]
	add	x29, sp, #32
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	wzr, [x29, #-4]
	stur	w0, [x29, #-8]
	str	x1, [sp, #16]
.Ltmp2:
	.loc	1 5 7 prologue_end
	ldur	w8, [x29, #-8]
	cmp	w8, #1
	b.ne	.LBB2_2
	.loc	1 0 7 is_stmt 0
	adrp	x8, _Z1av
	add	x8, x8, :lo12:_Z1av
	.loc	1 6 9 is_stmt 1
	str	x8, [sp, #8]
	.loc	1 6 5 is_stmt 0
	b	.LBB2_3
.LBB2_2:
	.loc	1 0 5
	adrp	x8, _Z1bv
	add	x8, x8, :lo12:_Z1bv
	.loc	1 8 9 is_stmt 1
	str	x8, [sp, #8]
.LBB2_3:
	.loc	1 0 9 is_stmt 0
	adrp	x8, .L.cfi.jumptable
	add	x9, x8, :lo12:.L.cfi.jumptable
	.loc	1 9 10 is_stmt 1
	ldr	x8, [sp, #8]
	sub	x9, x8, x9
	lsr	x10, x9, #2
	orr	x9, x10, x9, lsl #62
	cmp	x9, #1
	b.ls	.LBB2_5
	brk	#0x1
.LBB2_5:
	blr	x8
	.loc	1 9 3 is_stmt 0
	ldp	x29, x30, [sp, #32]
	add	sp, sp, #48
	ret
.Ltmp3:
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc

	.p2align	2
	.type	.L.cfi.jumptable,@function
.L.cfi.jumptable:
.Lfunc_begin3:
	.cfi_startproc
	//APP
	b	_Z1av.cfi
	b	_Z1bv.cfi

	//NO_APP
.Lfunc_end3:
	.size	.L.cfi.jumptable, .Lfunc_end3-.L.cfi.jumptable
	.cfi_endproc

	.type	.L__unnamed_1,@object
	.section	.rodata,"a",@progbits
	.p2align	2
.L__unnamed_1:
	.size	.L__unnamed_1, 0

	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 7.0.0 (trunk 335774) (llvm/trunk 335775)"
.Linfo_string1:
	.asciz	"tiny.cc"
.Linfo_string2:
	.asciz	""
	.section	.debug_abbrev,"",@progbits
	.byte	1
	.byte	17
	.byte	0
	.byte	37
	.byte	14
	.byte	19
	.byte	5
	.byte	3
	.byte	14
	.byte	16
	.byte	23
	.byte	27
	.byte	14
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	38
	.hword	4
	.word	.debug_abbrev
	.byte	8
	.byte	1
	.word	.Linfo_string0
	.hword	4
	.word	.Linfo_string1
	.word	.Lline_table_start0
	.word	.Linfo_string2
	.xword	.Lfunc_begin0
	.word	.Lfunc_end2-.Lfunc_begin0
	.section	.debug_ranges,"",@progbits
	.section	.debug_macinfo,"",@progbits
	.byte	0

	.type	_Z1av,@function
.set _Z1av, .L.cfi.jumptable
	.type	_Z1bv,@function
.set _Z1bv, .L.cfi.jumptable+4
	.ident	"clang version 7.0.0 (trunk 335774) (llvm/trunk 335775)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
