# REQUIRES: asserts
# REQUIRES: system-linux
# RUN: llvm-mc -triple=systemz-unknown-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec -phony-externals -debug-only=jitlink %t 2>&1 | \
# RUN:   FileCheck %s
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
	.file	"exceptions.cpp"
                                        # Start of file scope inline assembly
	.globl	_ZSt21ios_base_library_initv

                                        # End of file scope inline assembly
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	stmg	%r11, %r15, 88(%r15)
	.cfi_offset %r11, -72
	.cfi_offset %r14, -48
	.cfi_offset %r15, -40
	aghi	%r15, -168
	.cfi_def_cfa_offset 328
	lgr	%r11, %r15
	.cfi_def_cfa_register %r11
	mvhi	164(%r11), 0
	lghi	%r2, 4
	brasl	%r14, __cxa_allocate_exception@PLT
	mvhi	0(%r2), 1
	lgrl	%r3, _ZTIi@GOT
	lghi	%r4, 0
	brasl	%r14, __cxa_throw@PLT
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __cxa_allocate_exception
	.addrsig_sym __cxa_throw
	.addrsig_sym _ZTIi
