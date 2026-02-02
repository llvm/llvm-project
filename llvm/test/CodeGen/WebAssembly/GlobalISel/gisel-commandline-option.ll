; RUN: llc -mtriple=wasm32-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -O0 -global-isel \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,NOFALLBACK,ENABLED-O0

; RUN: llc -mtriple=wasm32-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -global-isel \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,NOFALLBACK,ENABLED-O1

; RUN: llc -mtriple=wasm32-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -global-isel -global-isel-abort=2 \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,FALLBACK,ENABLED-O1

; RUN: llc -mtriple=wasm32-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 \
; RUN:   | FileCheck %s --check-prefixes=DISABLED

; ENABLED:       IRTranslator
; ENABLED-O0-NEXT:  Analysis containing CSE Info
; ENABLED-NEXT:  Analysis for ComputingKnownBits
; ENABLED-O1-NEXT:  MachineDominator Tree Construction
; ENABLED-O1-NEXT:  Analysis containing CSE Info
; ENABLED-O1-NEXT:  WebAssemblyPreLegalizerCombiner
; ENABLED-NEXT:  Legalizer
; ENABLED-O1-NEXT:  MachineDominator Tree Construction
; ENABLED-O1-NEXT:  WebAssemblyPostLegalizerCombiner
; ENABLED-NEXT:  RegBankSelect
; ENABLED-NEXT:  Analysis for ComputingKnownBits
; ENABLED-O1-NEXT:  Natural Loop Information
; ENABLED-O1-NEXT:  Lazy Branch Probability Analysis
; ENABLED-O1-NEXT:  Lazy Block Frequency Analysis
; ENABLED-NEXT:  InstructionSelect
; NOFALLBACK-NEXT:  WebAssembly Argument Move
; NOFALLBACK-NEXT:  WebAssembly Set p2align Operands
; NOFALLBACK-NEXT:  WebAssembly Fix br_table Defaults
; NOFALLBACK-NEXT:  WebAssembly Clean Code After Trap
; ENABLED-NEXT:  ResetMachineFunction

; FALLBACK:       WebAssembly Instruction Selection
; FALLBACK-NEXT:  WebAssembly Argument Move
; FALLBACK-NEXT:  WebAssembly Set p2align Operands
; FALLBACK-NEXT:  WebAssembly Fix br_table Defaults
; FALLBACK-NEXT:  WebAssembly Clean Code After Trap

; NOFALLBACK-NOT: WebAssembly Instruction Selection

; DISABLED-NOT: IRTranslator

; DISABLED: WebAssembly Instruction Selection
; DISABLED: Finalize ISel and expand pseudo-instructions

define void @empty() {
  ret void
}
