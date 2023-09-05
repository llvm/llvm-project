; RUN: llc -mtriple=riscv64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -O0 -global-isel \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix NOFALLBACK

; RUN: llc -mtriple=riscv64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -global-isel \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix NOFALLBACK  --check-prefix ENABLED-O1

; RUN: llc -mtriple=riscv64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -global-isel -global-isel-abort=2 \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix FALLBACK --check-prefix ENABLED-O1

; RUN: llc -mtriple=riscv64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix DISABLED

; ENABLED:       IRTranslator
; ENABLED-NEXT:  Analysis containing CSE Info
; ENABLED-NEXT:  Analysis for ComputingKnownBits
; ENABLED-NEXT:  Legalizer
; ENABLED-NEXT:  RegBankSelect
; ENABLED-NEXT:  Analysis for ComputingKnownBits
; ENABLED-O1-NEXT:  Lazy Branch Probability Analysis
; ENABLED-O1-NEXT:  Lazy Block Frequency Analysis
; ENABLED-NEXT:  InstructionSelect
; ENABLED-NEXT:  ResetMachineFunction

; FALLBACK:       RISC-V DAG->DAG Pattern Instruction Selection
; NOFALLBACK-NOT: RISC-V DAG->DAG Pattern Instruction Selection

; DISABLED-NOT: IRTranslator

; DISABLED: RISC-V DAG->DAG Pattern Instruction Selection
; DISABLED: Finalize ISel and expand pseudo-instructions

define void @empty() {
  ret void
}
