# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple thumbv7-windows-itanium -filetype obj -o %t/COFF_Thumb.o %s
# RUN: llvm-rtdyld -triple thumbv7-windows \
# RUN:   -dummy-extern ExitProcess=0x54769891 \
# RUN:   -dummy-extern unnamed_addr=0x00001024 -verify -check %s %t/COFF_Thumb.o

	.text
	.syntax unified

	.def unnamed_addr
		.scl 2
		.type 32
	.endef
	.global unnamed_addr

	.def branch24t
		.scl 2
		.type 32
	.endef
	.global branch24t
	.p2align 1
	.code 16
	.thumb_func
branch24t:
@ rel1:
#	b unnamed_addr					@ IMAGE_REL_ARM_BRANCH24T

	.def function
		.scl 2
		.type 32
	.endef
	.globl function
	.p2align 1
	.code 16
	.thumb_func
function:
	push.w {r11, lr}
	mov r11, sp
rel3:							@ IMAGE_REL_ARM_MOV32T
	movw r0, :lower16:string
# rtdyld-check: decode_operand(rel3, 1) = (string&0x0000ffff)
	movt r0, :upper16:string
# TODO rtdyld-check: decode_operand(rel3, 1) = (string&0xffff0000>>16)
	blx r1
load_from_dllimport_lo:							@ IMAGE_REL_ARM_MOV32T
	movw r0, :lower16:__imp_ExitProcess
# (1) Check stub content.
# rtdyld-check: *{4}(stub_addr(COFF_Thumb.o/.text, __imp_ExitProcess)) = \
# rtdyld-check:   ExitProcess
#
# (2) Check lo bits of stub address.
# rtdyld-check: decode_operand(load_from_dllimport_lo, 1) = \
# rtdyld-check:   stub_addr(COFF_Thumb.o/.text, __imp_ExitProcess)[15:0]
load_from_dllimport_hi:
	movt r0, :upper16:__imp_ExitProcess
# (3) Check hi bits of stub address.
# rtdyld-check: decode_operand(load_from_dllimport_hi, 2) = \
# rtdyld-check:   stub_addr(COFF_Thumb.o/.text, __imp_ExitProcess)[31:16]
	ldr r1, [r0]
	movs r0, #0
	pop.w {r11, lr}
	bx r1

	.def main
		.scl 2
		.type 32
	.endef
	.globl main
	.p2align 1
	.code 16
	.thumb_func
main:
	push.w {r11, lr}
	mov r11, sp
rel5:
#	bl function					@ IMAGE_REL_ARM_BLX23T
	movs r0, #0
	pop.w {r11, pc}

	.section .rdata,"dr"
	.global string
string:
	.asciz "Hello World\n"

	.data

	.global relocations
relocations:
@ rel8:
	.long function(imgrel)				@ IMAGE_REL_ARM_ADDR32NB
# rtdyld-check: *{4}relocations = function - section_addr(COFF_Thumb.o, .text)
rel9:
	.secidx __imp_ExitProcess		@ IMAGE_REL_ARM_SECTION
# rtdyld-check: *{2}rel9 = 2
rel10:
	.long relocations(secrel32)			@ IMAGE_REL_ARM_SECREL
# rtdyld-check: *{4}rel10 = relocations - section_addr(COFF_Thumb.o, .data)
rel11:
	.secrel32 relocations				@ IMAGE_REL_ARM_SECREL
# rtdyld-check: *{4}rel11 = relocations - section_addr(COFF_Thumb.o, .data)
	bx r0
	trap

	.text

	.p2align 2

branch_to_thumb_func:
@ rel14:                                                @ IMAGE_REL_ARM_MOV32T
        movw r0, :lower16:function
# rtdyld-check: decode_operand(branch_to_thumb_func, 1) = (function&0x0000ffff|1)
        movt r0, :upper16:function
# TODO rtdyld-check: decode_operand(branch_to_thumb_func, 1) = (function&0xffff0000>>16)
        bx r0
        trap

        .data

        .p2align 2
a_data_symbol:
        .long 1073741822

        .p2align 2

        .text

ref_to_data_symbol_addr:
@ rel15:                                                @ IMAGE_REL_ARM_MOV32T
        movw r0, :lower16:a_data_symbol
# rtdyld-check: decode_operand(ref_to_data_symbol_addr, 1) = (a_data_symbol&0x0000ffff)
        movt r0, :upper16:a_data_symbol
# TODO rtdyld-check: decode_operand(ref_to_data_symbol_addr, 1) = (a_data_symbol&0xffff0000>>16)
