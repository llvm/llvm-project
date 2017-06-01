; RUN: llvm-mc -filetype=obj -triple=avr %s -mattr=avr6 | llvm-objdump -r - | FileCheck %s

; Checks that a global symbol with the address of another
; symbol generates a R_AVR_16_PM relocation, as the symbol
; will always be in program memory.

; CHECK: RELOCATION RECORDS FOR [.rela.text]:
; CHECK-NEXT: 00000002 R_AVR_16_PM .text

foo:
	ret

.globl	ptr
ptr:
	.short	foo

