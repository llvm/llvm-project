; RUN: llc -mtriple=riscv64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -O0 -global-isel \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,NOFALLBACK,ENABLED-O0

; RUN: llc -mtriple=riscv64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -global-isel \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,NOFALLBACK,ENABLED-O1

; RUN: llc -mtriple=riscv64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -global-isel -global-isel-abort=2 \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,FALLBACK,ENABLED-O1

; RUN: llc -mtriple=riscv64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 \
; RUN:   | FileCheck %s --check-prefixes=DISABLED

; ENABLED:       IRTranslator
; ENABLED-NEXT:  Analysis for ComputingKnownBits
; ENABLED-O0-NEXT:  RISCVO0PreLegalizerCombiner
; ENABLED-O1-NEXT:  MachineDominator Tree Construction
; ENABLED-NEXT:  Analysis containing CSE Info
; ENABLED-O1-NEXT:  RISCVPreLegalizerCombiner
; ENABLED-NEXT:  Legalizer
; ENABLED-O1-NEXT:  MachineDominator Tree Construction
; ENABLED-O1-NEXT:  RISCVPostLegalizerCombiner
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
