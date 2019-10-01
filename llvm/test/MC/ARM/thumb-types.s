@ RUN: llvm-mc -triple armv7-elf -filetype obj -o - %s | llvm-readobj --symbols \
@ RUN:    | FileCheck %s

	.syntax unified

	.thumb

	.type implicit_function,%function
implicit_function:
	nop

	.type implicit_data,%object
implicit_data:
	.long 0

	.arm
	.type arm_function,%function
arm_function:
	nop

	.thumb

	.text

untyped_text_label:
	nop

	.type explicit_function,%function
explicit_function:
	nop

	.long	tls(TPOFF)

	.type indirect_function,%gnu_indirect_function
indirect_function:
	nop

	.data

untyped_data_label:
	nop

	.type explicit_data,%object
explicit_data:
	.long 0

	.section	.tdata,"awT",%progbits
	.type	tls,%object
	.align	2
tls:
	.long	42
	.size	tls, 4


@ CHECK: Symbol {
@ CHECK:   Name: arm_function
@ CHECK:   Value: 0x6
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: explicit_data
@ CHECK:   Value: 0x2
@ CHECK:   Type: Object
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: explicit_function
@ CHECK:   Value: 0xD
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: implicit_data
@ CHECK:   Value: 0x2
@ CHECK:   Type: Object
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: implicit_function
@ CHECK:   Value: 0x1
@ CHECK:   Type: Function
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: indirect_function
@ CHECK:   Value: 0x13
@ CHECK:   Type: GNU_IFunc
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: tls
@ CHECK:   Value: 0x0
@ CHECK:   Type: TLS
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: untyped_data_label
@ CHECK:   Value: 0x0
@ CHECK:   Type: None
@ CHECK: }

@ CHECK: Symbol {
@ CHECK:   Name: untyped_text_label
@ CHECK:   Value: 0xA
@ CHECK:   Type: None
@ CHECK: }

