@ RUN: llvm-mc -triple arm-eabi -show-encoding %s | FileCheck %s

	.syntax unified
	.text
	.arm

undefined:
	udf #0

@ CHECK: udf	#0                      @ encoding: [0xf0,0x00,0xf0,0xe7]

