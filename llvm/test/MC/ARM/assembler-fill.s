// RUN: llvm-mc --triple=thumbv7eb-linux-gnueabihf %s -filetype=obj | llvm-objdump -triple=thumbv7eb-linux-gnueabihf -s - | FileCheck %s

// CHECK: Contents of section .text
// CHECK-NEXT: 0000 d000bf00 
	
//  Make sure we emit in correct endianness.
	
// CHECK: Contents of section .data
// CHECK-NEXT:  0000 12341234 1234 

	.syntax unified
        .text
        .thumb
	.thumb_func
.L1:
        beq Label
.L2:
	nop
Label:

	.data
	.short 0x1234
	.fill (.L2 - .L1), 2, 0x1234
