@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi | FileCheck -check-prefix=ASM %s
@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi -filetype=obj -o %t.o
@ RUN: llvm-objdump -d -r %t.o -triple=armv7-linux-gnueabi | FileCheck -check-prefix=OBJ %s
@ RUN: llvm-mc %s -triple=thumbv7-linux-gnueabi -filetype=obj -o %t.o
@ RUN: llvm-objdump -d -r %t.o -triple=thumbv7-linux-gnueabi | FileCheck -check-prefix=THUMB %s

	.syntax unified
	.text
	.globl	barf
	.align	2
	.type	barf,%function
barf:                                   @ @barf
@ %bb.0:                                @ %entry
	movw	r0, :lower16:GOT-(.LPC0_2+8)
	movt	r0, :upper16:GOT-(.LPC0_2+8)
.LPC0_2:
	movw	r0, :lower16:extern_symbol+1234
	movt	r0, :upper16:extern_symbol+1234

	movw	r0, :lower16:(foo - bar + 1234)
	movt	r0, :upper16:(foo - bar + 1234)
foo:
bar:

@ ASM:          movw    r0, :lower16:(GOT-(.LPC0_2+8))
@ ASM-NEXT:     movt    r0, :upper16:(GOT-(.LPC0_2+8))
@ ASM:          movw    r0, :lower16:(extern_symbol+1234)
@ ASM-NEXT:     movt    r0, :upper16:(extern_symbol+1234)
@ ASM:          movw    r0, :lower16:((foo-bar)+1234)
@ ASM-NEXT:     movt    r0, :upper16:((foo-bar)+1234)

@OBJ:      Disassembly of section .text:
@OBJ-NEXT: barf:
@OBJ-NEXT: 0:             f0 0f 0f e3     movw    r0, #65520
@OBJ-NEXT: 00000000:         R_ARM_MOVW_PREL_NC   GOT
@OBJ-NEXT: 4:             f4 0f 4f e3     movt    r0, #65524
@OBJ-NEXT: 00000004:         R_ARM_MOVT_PREL      GOT
@OBJ-NEXT: 8:             d2 04 00 e3     movw    r0, #1234
@OBJ-NEXT: 00000008:         R_ARM_MOVW_ABS_NC    extern_symbol
@OBJ-NEXT: c:             d2 04 40 e3     movt    r0, #1234
@OBJ-NEXT: 0000000c:         R_ARM_MOVT_ABS       extern_symbol
@OBJ-NEXT: 10:            d2 04 00 e3     movw    r0, #1234
@OBJ-NEXT: 14:            00 00 40 e3     movt    r0, #0

@THUMB:      Disassembly of section .text:
@THUMB-NEXT: barf:
@THUMB-NEXT: 0:             4f f6 f0 70     movw    r0, #65520
@THUMB-NEXT: 00000000:         R_ARM_THM_MOVW_PREL_NC GOT
@THUMB-NEXT: 4:             cf f6 f4 70     movt    r0, #65524
@THUMB-NEXT: 00000004:         R_ARM_THM_MOVT_PREL    GOT
@THUMB-NEXT: 8:             40 f2 d2 40     movw    r0, #1234
@THUMB-NEXT: 00000008:         R_ARM_THM_MOVW_ABS_NC  extern_symbol
@THUMB-NEXT: c:             c0 f2 d2 40     movt    r0, #1234
@THUMB-NEXT: 0000000c:         R_ARM_THM_MOVT_ABS     extern_symbol
@THUMB-NEXT: 10:            40 f2 d2 40     movw    r0, #1234
@THUMB-NEXT: 14:            c0 f2 00 00     movt    r0, #0
