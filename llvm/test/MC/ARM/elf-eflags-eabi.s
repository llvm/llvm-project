@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi -filetype=obj -o - | \
@ RUN:    elf-dump --dump-section-data  | FileCheck -check-prefix=OBJ %s
	.syntax unified
	.text
	.globl	barf
	.align	2
	.type	barf,%function
barf:                                   @ @barf
@ BB#0:                                 @ %entry
        b foo

@@@ make sure the EF_ARM_EABIMASK comes out OK
@OBJ:    'e_flags', 0x05000000
