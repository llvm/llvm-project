@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi | FileCheck -check-prefix=ASM %s
@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi -filetype=obj -o - | \
@ RUN:    elf-dump --dump-section-data | FileCheck -check-prefix=OBJ %s
	.syntax unified
	.text
	.globl	barf
	.align	2
	.type	barf,%function
barf:                                   @ @barf
@ BB#0:                                 @ %entry
	movw	r0, :lower16:GOT-(.LPC0_2+8)
	movt	r0, :upper16:GOT-(.LPC0_2+8)
.LPC0_2:
@ ASM:          movw    r0, :lower16:(GOT-(.LPC0_2+8))
@ ASM-NEXT:     movt    r0, :upper16:(GOT-(.LPC0_2+8))

@@ make sure that the text section fixups are sane too
@ OBJ:                 '.text'
@ OBJ-NEXT:            'sh_type', 0x00000001
@ OBJ-NEXT:            'sh_flags', 0x00000006
@ OBJ-NEXT:            'sh_addr', 0x00000000
@ OBJ-NEXT:            'sh_offset', 0x00000034
@ OBJ-NEXT:            'sh_size', 0x00000008
@ OBJ-NEXT:            'sh_link', 0x00000000
@ OBJ-NEXT:            'sh_info', 0x00000000
@ OBJ-NEXT:            'sh_addralign', 0x00000004
@ OBJ-NEXT:            'sh_entsize', 0x00000000
@ OBJ-NEXT:            '_section_data', 'f00f0fe3 f40f4fe3'

@ OBJ:              Relocation 0
@ OBJ-NEXT:         'r_offset', 0x00000000
@ OBJ-NEXT:         'r_sym'
@ OBJ-NEXT:         'r_type', 0x2d

@ OBJ:              Relocation 1
@ OBJ-NEXT:         'r_offset', 0x00000004
@ OBJ-NEXT:         'r_sym'
@ OBJ-NEXT:         'r_type', 0x2e

