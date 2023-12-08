@ RUN: llvm-mc -triple thumbv6m-eabi -show-encoding %s | FileCheck %s

	.syntax unified
	.text
	.thumb

undefined:
	udf #0

@ CHECK: udf	#0                      @ encoding: [0x00,0xde]

