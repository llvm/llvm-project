@ RUN: llvm-mc -triple thumbv7-eabi -mattr +thumb2 -show-encoding %s | FileCheck %s

	.syntax unified
	.text
	.thumb

undefined:
	udf #0
	udf.w #0

@ CHECK: udf	#0                      @ encoding: [0x00,0xde]
@ CHECK: udf.w	#0                      @ encoding: [0xf0,0xf7,0x00,0xa0]

