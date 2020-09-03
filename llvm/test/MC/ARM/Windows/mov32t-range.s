@ RUN: llvm-mc -triple thumbv7-windows-itanium -filetype obj -o - %s \
@ RUN:   | llvm-readobj -r - | FileCheck -check-prefix CHECK-RELOCATIONS %s

@ RUN: llvm-mc -triple thumbv7-windows-itanium -filetype obj -o - %s \
@ RUN:   | llvm-objdump -d - | FileCheck --check-prefix=CHECK-ENCODING %s

	.syntax unified
	.thumb
	.text

	.def truncation
		.scl 3
		.type 32
	.endef
	.align 2
	.thumb_func
truncation:
	movw r0, :lower16:.Lerange
	movt r0, :upper16:.Lerange
	bx lr

	.section .rdata,"rd"
.Lbuffer:
	.zero 65536
.Lerange:
	.asciz "-erange"

@ CHECK-RELOCATIONS: Relocations [
@ CHECK-RELOCATIONS:   .text {
@ CHECK-RELOCATIONS:     0x0 IMAGE_REL_ARM_MOV32T .rdata
@ CHECK-RELOCATIONS-NOT: 0x4 IMAGE_REL_ARM_MOV32T .rdata
@ CHECK-RELOCATIONS:   }
@ CHECK-RELOCATIONS: ]

@ CHECK-ENCODING:      0: 40 f2 00 00
@ CHECK-ENCODING-NEXT: 4: c0 f2 01 00

