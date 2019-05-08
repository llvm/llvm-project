@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s | llvm-readobj -S --sd \
@ RUN:   | FileCheck %s

	.syntax unified

	.text

	.even

	.global aligned_function
	.type aligned_function,%function
aligned_function:
	bkpt

	.space 5

	.even

	.global unaligned_function
	.type unaligned_function,%function
unaligned_function:
	bkpt

@ CHECK: Section {
@ CHECK:   Name: .text
@ CHECK:   SectionData (
@ CHECK:     0000: 700020E1 00000000 00007000 20E1
@ CHECK:   )
@ CHECK: }

	.data

	.space 15

	.even

	.global classifiable
	.type classifiable,%object
classifiable:
	.byte 0xf1
	.byte 0x51
	.byte 0xa5
	.byte 0xc1
	.byte 0x00
	.byte 0x00
	.byte 0x1e
	.byte 0xab

	.even

	.global declassified
	.type declassified,%object
declassified:
	.byte 0x51
	.byte 0xa5
	.byte 0xc1
	.byte 0xde
	.byte 0x00
	.byte 0x00
	.byte 0xed
	.byte 0xf1

@ CHECK: Section {
@ CHECK:   Name: .data
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 00000000 00000000 00000000
@ CHECK:     0010: F151A5C1 00001EAB 51A5C1DE 0000EDF1
@ CHECK:   )
@ CHECK: }

