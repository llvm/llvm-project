# REQUIRES: asserts
# RUN: llvm-mc -triple=arm64-apple-darwin11 -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec -phony-externals -debug-only=jitlink %t 2>&1 | \
# RUN:   FileCheck %s
#
# Check that splitting of eh-frame sections works.
#
# CHECK: DWARFRecordSectionSplitter: Processing __TEXT,__eh_frame...
# CHECK:  Processing block at
# CHECK:    Processing CFI record at
# CHECK:      Extracted {{.*}} section = __TEXT,__eh_frame
# CHECK:    Processing CFI record at
# CHECK:      Extracted {{.*}} section = __TEXT,__eh_frame
# CHECK: EHFrameEdgeFixer: Processing __TEXT,__eh_frame...
# CHECK:   Processing block at
# CHECK:     Processing CFI record at
# CHECK:       Record is CIE
# CHECK:   Processing block at
# CHECK:     Processing CFI record at
# CHECK:       Record is FDE
# CHECK:         Adding edge at {{.*}} to CIE at: {{.*}}
# CHECK:         Existing edge at {{.*}} to PC begin at {{.*}}
# CHECK:         Adding keep-alive edge from target at {{.*}} to FDE at {{.*}}
# CHECK:         Existing edge at {{.*}} to LSDA at {{.*}}

	.section	__TEXT,__text,regular,pure_instructions
 	.globl	_main
	.p2align	2
_main:
Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception0

	stp	x20, x19, [sp, #-32]!
	stp	x29, x30, [sp, #16]
	.cfi_def_cfa_offset 32
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	w0, #4
	bl	___cxa_allocate_exception
	mov	w8, #42
	str	w8, [x0]
Ltmp0:
Lloh0:
	adrp	x1, __ZTIi@GOTPAGE
Lloh1:
	ldr	x1, [x1, __ZTIi@GOTPAGEOFF]
	mov	x2, #0
	bl	___cxa_throw
Ltmp1:

	brk	#0x1
LBB0_2:
Ltmp2:
	bl	___cxa_begin_catch
	ldr	w19, [x0]
	bl	___cxa_end_catch
	mov	x0, x19
	ldp	x29, x30, [sp, #16]
	ldp	x20, x19, [sp], #32
	ret
	.loh AdrpLdrGot	Lloh0, Lloh1
Lfunc_end0:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table0:
Lexception0:
	.byte	255
	.byte	155
	.uleb128 Lttbase0-Lttbaseref0
Lttbaseref0:
	.byte	1
	.uleb128 Lcst_end0-Lcst_begin0
Lcst_begin0:
	.uleb128 Lfunc_begin0-Lfunc_begin0
	.uleb128 Ltmp0-Lfunc_begin0
	.byte	0
	.byte	0
	.uleb128 Ltmp0-Lfunc_begin0
	.uleb128 Ltmp1-Ltmp0
	.uleb128 Ltmp2-Lfunc_begin0
	.byte	1
	.uleb128 Ltmp1-Lfunc_begin0
	.uleb128 Lfunc_end0-Ltmp1
	.byte	0
	.byte	0
Lcst_end0:
	.byte	1

	.byte	0
	.p2align	2

Ltmp3:
	.long	__ZTIi@GOT-Ltmp3
Lttbase0:
	.p2align	2

.subsections_via_symbols
