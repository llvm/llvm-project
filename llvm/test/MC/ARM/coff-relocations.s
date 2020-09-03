@ RUN: llvm-mc -triple thumbv7-windows-itanium -filetype obj -o - %s \
@ RUN:   | llvm-readobj -r - | FileCheck %s -check-prefix CHECK-RELOCATION

@ RUN: llvm-mc -triple thumbv7-windows-itanium -filetype obj -o - %s \
@ RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-ENCODING

	.syntax unified
	.text
	.thumb

	.global target

	.thumb_func
branch24t_0:
	b target

@ CHECK-ENCODING-LABEL: <branch24t_0>:
@ CHECK-ENCODING-NEXT: b.w #0

	.thumb_func
branch24t_1:
	bl target

@ CHECK-ENCODING-LABEL: <branch24t_1>:
@ CHECK-ENCODING-NEXR: bl #0

	.thumb_func
branch20t:
	bcc target

@ CHECK-ENCODING-LABEL: <branch20t>:
@ CHECK-ENCODING-NEXT: blo.w #0

	.thumb_func
blx23t:
	blx target

@ CHECK-ENCODING-LABEL: <blx23t>:
@ CHECK-ENCODING-NEXT: blx #0

	.thumb_func
mov32t:
	movw r0, :lower16:target
	movt r0, :upper16:target
	blx r0

@ CHECK-ENCODING-LABEL: <mov32t>:
@ CHECK-ENCODING-NEXT: movw r0, #0
@ CHECK-ENCODING-NEXT: movt r0, #0
@ CHECK-ENCODING-NEXT: blx r0

	.thumb_func
addr32:
	ldr r0, .Laddr32
	bx r0
	trap
.Laddr32:
	.long target

@ CHECK-ENCODING-LABEL: <addr32>:
@ CHECK-ENCODING-NEXT: ldr r0, [pc, #4]
@ CHECK-ENCODING-NEXT: bx r0
@ CHECK-ENCODING-NEXT: trap
@ CHECK-ENCODING-NEXT: movs r0, r0
@ CHECK-ENCODING-NEXT: movs r0, r0

	.thumb_func
addr32nb:
	ldr r0, .Laddr32nb
	bx r0
	trap
.Laddr32nb:
	.long target(imgrel)

@ CHECK-ENCODING-LABEL: <addr32nb>:
@ CHECK-ENCODING-NEXT: ldr.w r0, [pc, #4]
@ CHECK-ENCODING-NEXT: bx r0
@ CHECK-ENCODING-NEXT: trap
@ CHECK-ENCODING-NEXT: movs r0, r0
@ CHECK-ENCODING-NEXT: movs r0, r0

       .thumb_func
secrel:
	ldr r0, .Lsecrel
	bx r0
	trap
.Lsecrel:
	.long target(secrel32)

@ CHECK-ENCODING-LABEL: <secrel>:
@ CHECK-ENCODING-NEXT: ldr.w r0, [pc, #4]
@ CHECK-ENCODING-NEXT: bx r0
@ CHECK-ENCODING-NEXT: trap
@ CHECK-ENCODING-NEXT: movs r0, r0
@ CHECK-ENCODING-NEXT: movs r0, r0

@ CHECK-RELOCATION: Relocations [
@ CHECK-RELOCATION:   Section (1) .text {
@ CHECK-RELOCATION:     0x0 IMAGE_REL_ARM_BRANCH24T
@ CHECK-RELOCATION:     0x4 IMAGE_REL_ARM_BRANCH24T
@ CHECK-RELOCATION:     0x8 IMAGE_REL_ARM_BRANCH20T
@ CHECK-RELOCATION:     0xC IMAGE_REL_ARM_BLX23T
@ CHECK-RELOCATION:     0x10 IMAGE_REL_ARM_MOV32T
@ CHECK-RELOCATION:     0x20 IMAGE_REL_ARM_ADDR32
@ CHECK-RELOCATION:     0x2C IMAGE_REL_ARM_ADDR32NB
@ CHECK-RELOCATION:     0x38 IMAGE_REL_ARM_SECREL
@ CHECK-RELOCATION:   }
@ CHECK-RELOCATION: ]

