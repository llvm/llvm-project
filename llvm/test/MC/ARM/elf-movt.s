@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi | FileCheck -check-prefix=ASM %s
@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi -filetype=obj -o %t.o
@ RUN: llvm-objdump --no-print-imm-hex -d -r %t.o --triple=armv7-linux-gnueabi | FileCheck --check-prefix=OBJ %s
@ RUN: llvm-mc %s -triple=thumbv7-linux-gnueabi -filetype=obj -o %t.o
@ RUN: llvm-objdump --no-print-imm-hex -d -r %t.o --triple=thumbv7-linux-gnueabi | FileCheck --check-prefix=THUMB %s

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
@OBJ-EMPTY:
@OBJ-NEXT: <barf>:
@OBJ-NEXT: 0:             e30f0ff0        movw    r0, #65520
@OBJ-NEXT: 00000000:         R_ARM_MOVW_PREL_NC   GOT
@OBJ-NEXT: 4:             e34f0ff4        movt    r0, #65524
@OBJ-NEXT: 00000004:         R_ARM_MOVT_PREL      GOT
@OBJ-NEXT: 8:             e30004d2        movw    r0, #1234
@OBJ-NEXT: 00000008:         R_ARM_MOVW_ABS_NC    extern_symbol
@OBJ-NEXT: c:             e34004d2        movt    r0, #1234
@OBJ-NEXT: 0000000c:         R_ARM_MOVT_ABS       extern_symbol
@OBJ-NEXT: 10:            e30004d2        movw    r0, #1234
@OBJ-NEXT: 14:            e3400000        movt    r0, #0

@THUMB:      Disassembly of section .text:
@THUMB-EMPTY:
@THUMB-NEXT: <barf>:
@THUMB-NEXT: 0:             f64f 70f0       movw    r0, #65520
@THUMB-NEXT: 00000000:         R_ARM_THM_MOVW_PREL_NC GOT
@THUMB-NEXT: 4:             f6cf 70f4       movt    r0, #65524
@THUMB-NEXT: 00000004:         R_ARM_THM_MOVT_PREL    GOT
@THUMB-NEXT: 8:             f240 40d2       movw    r0, #1234
@THUMB-NEXT: 00000008:         R_ARM_THM_MOVW_ABS_NC  extern_symbol
@THUMB-NEXT: c:             f2c0 40d2       movt    r0, #1234
@THUMB-NEXT: 0000000c:         R_ARM_THM_MOVT_ABS     extern_symbol
@THUMB-NEXT: 10:            f240 40d2       movw    r0, #1234
@THUMB-NEXT: 14:            f2c0 0000       movt    r0, #0
