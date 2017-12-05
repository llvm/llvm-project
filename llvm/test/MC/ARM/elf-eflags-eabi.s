@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi -filetype=obj -o - | \
@ RUN:    llvm-readobj -h | FileCheck -check-prefix=OBJ %s
	.syntax unified
	.text
	.globl	barf
	.align	2
	.type	barf,%function
barf:                                   @ @barf
@ %bb.0:                                @ %entry
        b foo

@@@ make sure the EF_ARM_EABIMASK comes out OK
@OBJ: ElfHeader {
@OBJ:   Flags [ (0x5000000)
