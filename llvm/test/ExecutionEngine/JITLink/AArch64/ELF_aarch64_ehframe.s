# REQUIRES: asserts
# RUN: llvm-mc -triple=aarch64-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec -phony-externals -debug-only=jitlink %t 2>&1 | \
# RUN:   FileCheck %s
#
# Check that splitting of eh-frame sections works.
#
# CHECK: DWARFRecordSectionSplitter: Processing .eh_frame...
# CHECK:  Processing block at
# CHECK:    Processing CFI record at
# CHECK:      Extracted {{.*}} section = .eh_frame
# CHECK:    Processing CFI record at
# CHECK:      Extracted {{.*}} section = .eh_frame
# CHECK: EHFrameEdgeFixer: Processing .eh_frame...
# CHECK:   Processing block at
# CHECK:     Processing CFI record at
# CHECK:       Record is CIE
# CHECK:   Processing block at
# CHECK:     Processing CFI record at
# CHECK:       Record is FDE
# CHECK:         Adding edge at {{.*}} to CIE at: {{.*}}
# CHECK:         Existing edge at {{.*}} to PC begin at {{.*}}
# CHECK:         Adding keep-alive edge from target at {{.*}} to FDE at {{.*}}
# CHECK:   Processing block at
# CHECK:     Processing CFI record at
# CHECK:       Record is FDE
# CHECK:         Adding edge at {{.*}} to CIE at: {{.*}}
# CHECK:         Existing edge at {{.*}} to PC begin at {{.*}}
# CHECK:         Adding keep-alive edge from target at {{.*}} to FDE at {{.*}}

	.text
	.globl	main
	.p2align	2
	.type	main,@function
main: 
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	wzr, [x29, #-4]
	mov	x0, #4
	bl	__cxa_allocate_exception
	mov	w8, #1
	str	w8, [x0]
	adrp	x1, :got:_ZTIi
	ldr	x1, [x1, :got_lo12:_ZTIi]
	mov	x2, xzr
	bl	__cxa_throw
.main_end:
	.size	main, .main_end-main
	.cfi_endproc

	.globl	dup
	.p2align	2
	.type	dup,@function
dup: 
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	wzr, [x29, #-4]
	mov	x0, #4
	bl	__cxa_allocate_exception
	mov	w8, #1
	str	w8, [x0]
	adrp	x1, :got:_ZTIi
	ldr	x1, [x1, :got_lo12:_ZTIi]
	mov	x2, xzr
	bl	__cxa_throw
.dup_end:
	.size	dup, .dup_end-dup
	.cfi_endproc