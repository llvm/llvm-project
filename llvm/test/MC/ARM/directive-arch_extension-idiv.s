@ RUN: not llvm-mc -triple armv6-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-ARMv6 -check-prefix CHECK-V6
@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-ARMv7 -check-prefix CHECK-V7
@ RUN: not llvm-mc -triple armv7m-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-ARMv7M -check-prefix CHECK-V7M
@ RUN: not llvm-mc -triple thumbv6-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-THUMBv6 -check-prefix CHECK-V6
@ RUN: not llvm-mc -triple thumbv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-THUMBv7 -check-prefix CHECK-V7
@ RUN: not llvm-mc -triple thumbv7m-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-THUMBv7M -check-prefix CHECK-V7M

	.syntax unified

	.arch_extension idiv
@ CHECK-V6: error: architectural extension 'idiv' is not allowed for the current base architecture
@ CHECK-V6-NEXT: 	.arch_extension idiv
@ CHECK-V6-NEXT:                     ^
@ CHECK-V7M: error: architectural extension 'idiv' is not allowed for the current base architecture
@ CHECK-V7M-NEXT: 	.arch_extension idiv
@ CHECK-V7M-NEXT:                     ^

	.type idiv,%function
idiv:
	udiv r0, r1, r2
@ CHECK-ARMv6: error: instruction requires: divide in ARM
@ CHECK-THUMBv6: error: instruction requires: divide in THUMB armv8m.base
	sdiv r0, r1, r2
@ CHECK-ARMv6: error: instruction requires: divide in ARM
@ CHECK-THUMBv6: error: instruction requires: divide in THUMB armv8m.base

	.arch_extension noidiv
@ CHECK-V6: error: architectural extension 'idiv' is not allowed for the current base architecture
@ CHECK-V6-NEXT: 	.arch_extension noidiv
@ CHECK-V6-NEXT:                     ^
@ CHECK-V7M: error: architectural extension 'idiv' is not allowed for the current base architecture
@ CHECK-V7M-NEXT: 	.arch_extension noidiv
@ CHECK-V7M-NEXT:                     ^

	.type noidiv,%function
noidiv:
	udiv r0, r1, r2
@ CHECK-ARMv6: error: instruction requires: divide in ARM
@ CHECK-THUMBv6: error: instruction requires: divide in THUMB armv8m.base
@ CHECK-ARMv7: error: instruction requires: divide in ARM
@ CHECK-THUMBv7: error: instruction requires: divide in THUMB
	sdiv r0, r1, r2
@ CHECK-ARMv6: error: instruction requires: divide in ARM
@ CHECK-THUMBv6: error: instruction requires: divide in THUMB armv8m.base
@ CHECK-ARMv7: error: instruction requires: divide in ARM
@ CHECK-THUMBv7: error: instruction requires: divide in THUMB

