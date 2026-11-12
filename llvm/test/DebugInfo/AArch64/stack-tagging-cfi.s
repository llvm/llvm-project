# REQUIRES: aarch64-registered-target

# RUN: llvm-mc -filetype=obj -triple aarch64-elfs -mattr=+mte %s -o %t.o
# RUN: llvm-dwarfdump --eh-frame %t.o | FileCheck %s

# CHECK: Augmentation:          "zRG"

	.text
	.globl	WithUnwind
	.p2align	2
	.type	WithUnwind,@function
WithUnwind:
	.cfi_startproc
	.cfi_mte_tagged_frame
	ret
.Lfunc_end0:
	.size	WithUnwind, .Lfunc_end0-WithUnwind
	.cfi_endproc
