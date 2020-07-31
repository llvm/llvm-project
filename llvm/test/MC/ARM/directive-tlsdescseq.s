@ RUN: llvm-mc -triple armv7-linux-gnu -filetype obj -o - %s \
@ RUN:   | llvm-readobj -r - \
@ RUN:   | FileCheck %s
@ RUN: llvm-mc -triple armv7-linux-gnu -filetype asm -o - %s \
@ RUN:   | FileCheck -check-prefix CHECK-ASM %s

	.type tlsdescseq,%function
tlsdescseq:
	ldr r1, [pc, #8]
1:
.tlsdescseq variable
	add r2, pc, r1
.tlsdescseq variable
	ldr r3, [r1, #4]
.tlsdescseq variable
	blx r3
2:
	.word variable(tlsdesc) + (. - 1b)

@ CHECK: Relocations [
@ CHECK:     0x4 R_ARM_TLS_DESCSEQ variable 0x0
@ CHECK:     0x8 R_ARM_TLS_DESCSEQ variable 0x0
@ CHECK:     0xC R_ARM_TLS_DESCSEQ variable 0x0
@ CHECK:     0x10 R_ARM_TLS_GOTDESC variable 0x0
@ CHECK: ]

@ CHECK-ASM: ldr r1, [pc, #8]
@ CHECK-ASM: .tlsdescseq variable
@ CHECK-ASM-NEXT: add r2, pc, r1
@ CHECK-ASM: .tlsdescseq variable
@ CHECK-ASM-NEXT: ldr r3, [r1, #4]
@ CHECK-ASM: .tlsdescseq variable
@ CHECK-ASM-NEXT: blx r3

