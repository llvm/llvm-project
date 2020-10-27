@@ test st_value bit 0 of thumb function
@ RUN: llvm-mc %s -triple=thumbv7-linux-gnueabi -filetype=obj -o - | \
@ RUN: llvm-readobj --symbols - | FileCheck %s
	.syntax unified
	.text
	.globl	foo
	.align	2
	.code	16
	.thumb_func
	.type	foo,%function
foo:
	bx	lr

	.global bar
bar = foo

	.global baz
baz = foo + 2

@@ make sure foo and bar are thumb function: bit 0 = 1 (st_value)
@CHECK:        Symbol {
@CHECK:          Name: bar
@CHECK-NEXT:     Value: 0x1
@CHECK-NEXT:     Size: 0
@CHECK-NEXT:     Binding: Global
@CHECK-NEXT:     Type: Function

@CHECK:        Symbol {
@CHECK:          Name: baz
@CHECK-NEXT:     Value: 0x3
@CHECK-NEXT:     Size: 0
@CHECK-NEXT:     Binding: Global
@CHECK-NEXT:     Type: Function

@CHECK:        Symbol {
@CHECK:          Name: foo
@CHECK-NEXT:     Value: 0x1
@CHECK-NEXT:     Size: 0
@CHECK-NEXT:     Binding: Global
@CHECK-NEXT:     Type: Function
