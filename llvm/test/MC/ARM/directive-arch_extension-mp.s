@ RUN: not llvm-mc -triple armv6-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V6
@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V7
@ RUN: not llvm-mc -triple armv7m-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V7M
@ RUN: not llvm-mc -triple thumbv6-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V6
@ RUN: not llvm-mc -triple thumbv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V7
@ RUN: not llvm-mc -triple thumbv7m-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V7M

	.syntax unified

	.arch_extension mp
@ CHECK-V6: error: architectural extension 'mp' is not allowed for the current base architecture
@ CHECK-V6-NEXT: 	.arch_extension mp
@ CHECK-V6-NEXT:                     ^

	.type mp,%function
mp:
	pldw [r0]
@ CHECK-V6: error: instruction requires: mp-extensions armv7
@ CHECK-V7M: error: instruction requires: mp-extensions

	.arch_extension nomp
@ CHECK-V6: error: architectural extension 'mp' is not allowed for the current base architecture
@ CHECK-V6-NEXT: 	.arch_extension nomp
@ CHECK-V6-NEXT:                     ^

	.type nomp,%function
nomp:
	pldw [r0]
@ CHECK-V6: error: instruction requires: mp-extensions armv7
@ CHECK-V7: error: instruction requires: mp-extensions
@ CHECK-V7M: error: instruction requires: mp-extensions

