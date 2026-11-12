# RUN: llvm-mc %s -filetype obj -triple armv7a-- -o %t.o
# RUN: not llvm-cfi-verify %t.o 2>&1 | FileCheck %s

# CHECK: Could not initialise disassembler: Unsupported architecture.

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
	.syntax unified
	.eabi_attribute	67, "2.09"
	.eabi_attribute	6, 2
	.eabi_attribute	8, 1
	.eabi_attribute	9, 1
	.eabi_attribute	34, 1
	.eabi_attribute	15, 1
	.eabi_attribute	16, 1
	.eabi_attribute	17, 2
	.eabi_attribute	20, 1
	.eabi_attribute	21, 1
	.eabi_attribute	23, 3
	.eabi_attribute	24, 1
	.eabi_attribute	25, 1
	.eabi_attribute	38, 1
	.eabi_attribute	18, 4
	.eabi_attribute	26, 2
	.eabi_attribute	14, 0
	.file	"ld-temp.o"
	.p2align	2
	.type	_Z1av.cfi,%function
	.code	32
_Z1av.cfi:
.Lfunc_begin0:
	.file	1 "tiny.cc"
	.loc	1 1 0
	.fnstart
	.cfi_sections .debug_frame
	.cfi_startproc
	.loc	1 1 11 prologue_end
	mov	r0, #42
	bx	lr
.Ltmp0:
.Lfunc_end0:
	.size	_Z1av.cfi, .Lfunc_end0-_Z1av.cfi
	.cfi_endproc
	.cantunwind
	.fnend

	.p2align	2
	.type	_Z1bv.cfi,%function
	.code	32
_Z1bv.cfi:
.Lfunc_begin1:
	.loc	1 2 0
	.fnstart
	.cfi_startproc
	.loc	1 2 11 prologue_end
	mov	r0, #137
	bx	lr
.Ltmp1:
.Lfunc_end1:
	.size	_Z1bv.cfi, .Lfunc_end1-_Z1bv.cfi
	.cfi_endproc
	.cantunwind
	.fnend

	.p2align	2
	.type	main,%function
	.code	32
main:
.Lfunc_begin2:
	.loc	1 3 0
	.fnstart
	.cfi_startproc
	.save	{r11, lr}
	push	{r11, lr}
	.cfi_def_cfa_offset 8
	.cfi_offset lr, -4
	.cfi_offset r11, -8
	.setfp	r11, sp
	mov	r11, sp
	.cfi_def_cfa_register r11
	.pad	#16
	sub	sp, sp, #16
	mov	r2, #0
	str	r2, [r11, #-4]
	str	r0, [sp, #8]
	str	r1, [sp, #4]
.Ltmp2:
	.loc	1 5 7 prologue_end
	ldr	r0, [sp, #8]
	cmp	r0, #1
	bne	.LBB2_2
	b	.LBB2_1
.LBB2_1:
	.loc	1 6 9
	ldr	r0, .LCPI2_0
.LPC2_0:
	add	r0, pc, r0
	str	r0, [sp]
	.loc	1 6 5 is_stmt 0
	b	.LBB2_3
.LBB2_2:
	.loc	1 8 9 is_stmt 1
	ldr	r0, .LCPI2_1
.LPC2_1:
	add	r0, pc, r0
	str	r0, [sp]
	b	.LBB2_3
.LBB2_3:
	.loc	1 9 10
	ldr	r1, [sp]
	ldr	r0, .LCPI2_2
.LPC2_2:
	add	r0, pc, r0
	sub	r0, r1, r0
	ror	r0, r0, #2
	cmp	r0, #2
	blo	.LBB2_5
	b	.LBB2_4
.LBB2_4:
	.inst	0xe7ffdefe
.LBB2_5:
	mov	lr, pc
	bx	r1
	.loc	1 9 3 is_stmt 0
	mov	sp, r11
	pop	{r11, lr}
	bx	lr
.Ltmp3:
	.p2align	2
	.loc	1 0 3
.LCPI2_0:
	.long	_Z1av-(.LPC2_0+8)
.LCPI2_1:
	.long	_Z1bv-(.LPC2_1+8)
.LCPI2_2:
	.long	.L.cfi.jumptable-(.LPC2_2+8)
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
	.fnend

	.p2align	2
	.type	.L.cfi.jumptable,%function
	.code	32
.L.cfi.jumptable:
.Lfunc_begin3:
	.fnstart
	.cfi_startproc
	@APP
	b	_Z1av.cfi
	b	_Z1bv.cfi

	@NO_APP
.Lfunc_end3:
	.size	.L.cfi.jumptable, .Lfunc_end3-.L.cfi.jumptable
	.cfi_endproc
	.cantunwind
	.fnend

	.type	.L__unnamed_1,%object
	.section	.rodata,"a",%progbits
.L__unnamed_1:
	.size	.L__unnamed_1, 0

	.section	.debug_str,"MS",%progbits,1
.Linfo_string0:
	.asciz	"clang version 7.0.0 (trunk 336681) (llvm/trunk 336683)"
.Linfo_string1:
	.asciz	"tiny.cc"
.Linfo_string2:
	.asciz	""
	.section	.debug_abbrev,"",%progbits
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
	.section	.debug_info,"",%progbits
.Lcu_begin0:
	.long	34
	.short	4
	.long	.debug_abbrev
	.byte	4
	.byte	1
	.long	.Linfo_string0
	.short	4
	.long	.Linfo_string1
	.long	.Lline_table_start0
	.long	.Linfo_string2
	.long	.Lfunc_begin0
	.long	.Lfunc_end2-.Lfunc_begin0
	.section	.debug_ranges,"",%progbits
	.section	.debug_macinfo,"",%progbits
	.byte	0

	.globl	__typeid__ZTSFivE_global_addr
	.hidden	__typeid__ZTSFivE_global_addr
.set __typeid__ZTSFivE_global_addr, .L.cfi.jumptable
	.size	__typeid__ZTSFivE_global_addr, 1
	.type	_Z1av,%function
.set _Z1av, .L.cfi.jumptable
	.type	_Z1bv,%function
.set _Z1bv, .L.cfi.jumptable+4
	.ident	"clang version 7.0.0 (trunk 336681) (llvm/trunk 336683)"
	.section	".note.GNU-stack","",%progbits
	.section	.debug_line,"",%progbits
.Lline_table_start0:
