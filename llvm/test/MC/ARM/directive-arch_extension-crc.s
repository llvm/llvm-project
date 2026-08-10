@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V7
@ RUN: not llvm-mc -triple armv8-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V8

	.syntax unified

	.arm

	.arch_extension crc
@ CHECK-V7: error: architectural extension 'crc' is not allowed for the current base architecture
@ CHECK-V7-NEXT: 	.arch_extension crc
@ CHECK-V7-NEXT:                     ^

	.type crc,%function
crc:
	crc32b r0, r1, r2
@ CHECK-V7: error: instruction requires: crc
	crc32h r0, r1, r2
@ CHECK-V7: error: instruction requires: crc
	crc32w r0, r1, r2
@ CHECK-V7: error: instruction requires: crc

	crc32cb r0, r1, r2
@ CHECK-V7: error: instruction requires: crc
	crc32ch r0, r1, r2
@ CHECK-V7: error: instruction requires: crc
	crc32cw r0, r1, r2
@ CHECK-V7: error: instruction requires: crc

	.arch_extension nocrc
@ CHECK-V7: error: architectural extension 'crc' is not allowed for the current base architecture
@ CHECK-V7-NEXT: 	.arch_extension nocrc
@ CHECK-V7-NEXT:                     ^

	.type nocrc,%function
nocrc:
	crc32b r0, r1, r2
@ CHECK-V7: error: instruction requires: crc
@ CHECK-V8: error: instruction requires: crc
	crc32h r0, r1, r2
@ CHECK-V7: error: instruction requires: crc
@ CHECK-V8: error: instruction requires: crc
	crc32w r0, r1, r2
@ CHECK-V7: error: instruction requires: crc
@ CHECK-V8: error: instruction requires: crc

	crc32cb r0, r1, r2
@ CHECK-V7: error: instruction requires: crc
@ CHECK-V8: error: instruction requires: crc
	crc32ch r0, r1, r2
@ CHECK-V7: error: instruction requires: crc
@ CHECK-V8: error: instruction requires: crc
	crc32cw r0, r1, r2
@ CHECK-V7: error: instruction requires: crc
@ CHECK-V8: error: instruction requires: crc

