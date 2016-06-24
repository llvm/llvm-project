; Test all important variants of the unconditional 'br' instruction.

; RUN: llc -march=mips   -mcpu=mips32   -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,NOT-R6
; RUN: llc -march=mips   -mcpu=mips32r2 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,NOT-R6
; RUN: llc -march=mips   -mcpu=mips32r3 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,NOT-R6
; RUN: llc -march=mips   -mcpu=mips32r5 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,NOT-R6
; RUN: llc -march=mips   -mcpu=mips32r6 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,R6C
; RUN: llc -march=mips64 -mcpu=mips4    -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,NOT-R6
; RUN: llc -march=mips64 -mcpu=mips64   -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,NOT-R6
; RUN: llc -march=mips64 -mcpu=mips64r2 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,NOT-R6
; RUN: llc -march=mips64 -mcpu=mips64r3 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,NOT-R6
; RUN: llc -march=mips64 -mcpu=mips64r5 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,NOT-R6
; RUN: llc -march=mips64 -mcpu=mips64r6 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,R6

define i32 @br(i8 *%addr) {
; ALL-LABEL: br:
; NOT-R6:        jr $4 # <MCInst #{{[0-9]+}} JR
; R6C:           jrc $4 # <MCInst #{{[0-9]+}} JIC


; ALL: $BB0_1: # %L1
; NOT-R6:        jr $ra # <MCInst #{{[0-9]+}} JR
; R6:            jr $ra # <MCInst #{{[0-9]+}} JALR
; R6C:           jr $ra # <MCInst #{{[0-9]+}} JALR
; ALL:           addiu $2, $zero, 0

; ALL: $BB0_2: # %L2
; NOT-R6:        jr $ra # <MCInst #{{[0-9]+}} JR
; R6:            jr $ra # <MCInst #{{[0-9]+}} JALR
; R6C:           jr $ra # <MCInst #{{[0-9]+}} JALR
; ALL:           addiu $2, $zero, 1

entry:
  indirectbr i8* %addr, [label %L1, label %L2]

L1:
  ret i32 0

L2:
  ret i32 1
}
