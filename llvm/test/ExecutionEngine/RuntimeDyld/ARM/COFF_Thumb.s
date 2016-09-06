// RUN: llvm-mc -triple thumbv7-windows-itanium -filetype obj -o %t.obj %s
// RUN: llvm-rtdyld -triple thumbv7-windows -dummy-extern OutputDebugStringA=0x78563412 -dummy-extern ExitProcess=0x54769890 -dummy-extern unnamed_addr=0x00001024 -verify -check %s %t.obj

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
rel2:							@ IMAGE_REL_ARM_MOV32T
	movw r0, :lower16:__imp_OutputDebugStringA
# rtdyld-check: decode_operand(rel2, 1) = (__imp_OutputDebugStringA&0x0000ffff)
	movt r0, :upper16:__imp_OutputDebugStringA
# TODO rtdyld-check: decode_operand(rel2, 1) = (__imp_OutputDebugStringA&0xffff0000>>16)
	ldr r1, [r0]
rel3:							@ IMAGE_REL_ARM_MOV32T
	movw r0, :lower16:string
# rtdyld-check: decode_operand(rel3, 1) = (string&0x0000ffff)
	movt r0, :upper16:string
# TODO rtdyld-check: decode_operand(rel3, 1) = (string&0xffff0000>>16)
	blx r1
rel4:							@ IMAGE_REL_ARM_MOV32T
	movw r0, :lower16:__imp_ExitProcess
# rtdyld-check: decode_operand(rel4, 1) = (__imp_ExitProcess&0x0000ffff)
	movt r0, :upper16:__imp_ExitProcess
# TODO rtdyld-check: decode_operand(rel4, 1) = (__imp_ExitProcess&0xffff0000>>16)
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

	.p2align 2
__imp_OutputDebugStringA:
@ rel6:
	.long OutputDebugStringA			@ IMAGE_REL_ARM_ADDR32
# rtdyld-check: *{4}__imp_OutputDebugStringA = 0x78563412

	.p2align 2
__imp_ExitProcess:
@ rel7:
	.long ExitProcess				@ IMAGE_REL_ARM_ADDR32
# rtdyld-check: *{4}__imp_ExitProcess = 0x54769890

	.global relocations
relocations:
@ rel8:
	.long function(imgrel)				@ IMAGE_REL_ARM_ADDR32NB
# rtdyld-check: *{4}relocations = function - section_addr(COFF_Thumb.s.tmp.obj, .text)
rel9:
	.secidx __imp_OutputDebugStringA		@ IMAGE_REL_ARM_SECTION
# rtdyld-check: *{2}rel9 = 1
rel10:
	.long relocations(secrel32)			@ IMAGE_REL_ARM_SECREL
# rtdyld-check: *{4}rel10 = relocations - section_addr(COFF_Thumb.s.tmp.obj, .data)
rel11:
	.secrel32 relocations				@ IMAGE_REL_ARM_SECREL
# rtdyld-check: *{4}rel11 = relocations - section_addr(COFF_Thumb.s.tmp.obj, .data)

