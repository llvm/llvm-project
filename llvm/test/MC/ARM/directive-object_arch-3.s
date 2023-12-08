@ RUN: llvm-mc -triple armv7-eabi -filetype asm -o - %s | FileCheck %s

	.syntax unified

	.arch armv7
	.object_arch armv4

@ CHECK: .text
@ CHECK: .arch	armv7
@ CHECK: .object_arch	armv4

