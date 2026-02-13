# REQUIRES: asserts
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec -phony-externals \
# RUN:              %t 2>&1 \
# RUN:              | FileCheck %s
# RUN: llvm-mc -triple=powerpc64-unknown-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec -phony-externals \
# RUN:              %t 2>&1 \
# RUN:              | FileCheck %s
#
# Check that splitting of eh-frame sections works.
#
# CHECK: DWARFRecordSectionSplitter: Processing .eh_frame...
# CHECK:   Processing block at
# CHECK:     Processing CFI record at
# CHECK:     Processing CFI record at
# CHECK: EHFrameEdgeFixer: Processing .eh_frame in "{{.*}}"...
# CHECK:   Processing block at
# CHECK:     Record is CIE
# CHECK:   Processing block at
# CHECK:     Record is FDE
# CHECK:       Adding edge at {{.*}} to CIE at: {{.*}}
# CHECK:       Processing PC-begin at
# CHECK:       Existing edge at {{.*}} to PC begin at {{.*}}
# CHECK:       Adding keep-alive edge from target at {{.*}} to FDE at {{.*}}

	.text
	.abiversion 2
	.file	"exception.cc"
	.globl	main
	.p2align	4
	.type	main,@function
main:
.Lfunc_begin0:
	.cfi_startproc
.Lfunc_gep0:
	addis 2, 12, .TOC.-.Lfunc_gep0@ha
	addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
	.localentry	main, .Lfunc_lep0-.Lfunc_gep0
	mflr 0
	stdu 1, -32(1)
	std 0, 48(1)
	.cfi_def_cfa_offset 32
	.cfi_offset lr, 16
	li 3, 8
	bl __cxa_allocate_exception
	nop
	addis 4, 2, .LC0@toc@ha
	addis 5, 2, .LC1@toc@ha
	addis 6, 2, .LC2@toc@ha
	ld 4, .LC0@toc@l(4)
	addi 4, 4, 16
	std 4, 0(3)
	ld 4, .LC1@toc@l(5)
	ld 5, .LC2@toc@l(6)
	bl __cxa_throw
	nop
	.long	0
	.quad	0
.Lfunc_end0:
	.size	main, .Lfunc_end0-.Lfunc_begin0
	.cfi_endproc

	.ident	"clang version 17.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _ZTISt9exception
	.section	.toc,"aw",@progbits
.LC0:
	.tc _ZTVSt9exception[TC],_ZTVSt9exception
.LC1:
	.tc _ZTISt9exception[TC],_ZTISt9exception
.LC2:
	.tc _ZNSt9exceptionD1Ev[TC],_ZNSt9exceptionD1Ev
