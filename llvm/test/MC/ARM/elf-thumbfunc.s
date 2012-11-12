@@ test st_value bit 0 of thumb function
@ RUN: llvm-mc %s -triple=thumbv7-linux-gnueabi -filetype=obj -o - | \
@ RUN: elf-dump  | FileCheck %s
	.syntax unified
	.text
	.globl	foo
	.align	2
	.type	foo,%function
	.code	16
	.thumb_func
foo:
	bx	lr

@@ make sure foo is thumb function: bit 0 = 1 (st_value)
@CHECK:           Symbol 4
@CHECK-NEXT:      'st_name', 0x00000001
@CHECK-NEXT:      'st_value', 0x00000001
@CHECK-NEXT:      'st_size', 0x00000000
@CHECK-NEXT:      'st_bind', 0x1
@CHECK-NEXT:      'st_type', 0x2
