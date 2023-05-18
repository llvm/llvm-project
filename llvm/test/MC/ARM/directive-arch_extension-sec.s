@ RUN: not llvm-mc -triple armv6-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V6
@ RUN: not llvm-mc -triple armv6k-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V7
@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V7
@ RUN: not llvm-mc -triple thumbv6-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V6
@ RUN: not llvm-mc -triple thumbv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V7

	.syntax unified

	.arch_extension sec
@ CHECK-V6: error: architectural extension 'sec' is not allowed for the current base architecture
@ CHECK-V6-NEXT: 	.arch_extension sec
@ CHECK-V6-NEXT:                     ^
@ CHECK-V7-NOT: error: architectural extension 'sec' is not allowed for the current base architecture

	.type sec,%function
sec:
	smc #0
@ CHECK-V6: error: instruction requires: TrustZone

	.arch_extension nosec
@ CHECK-V6: error: architectural extension 'sec' is not allowed for the current base architecture
@ CHECK-V6-NEXT: 	.arch_extension nosec
@ CHECK-V6-NEXT:                     ^
@ CHECK-V7-NOT: error: architectural extension 'sec' is not allowed for the current base architecture

	.type nosec,%function
nosec:
	smc #0
@ CHECK-V7: error: instruction requires: TrustZone
@ CHECK-V7-NOT: error: instruction requires: TrustZone

