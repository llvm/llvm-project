; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   --debugify-and-strip-all-safe=0 \
; RUN:   -verify-machineinstrs=0 -O0 \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,FALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   --debugify-and-strip-all-safe=0 \
; RUN:   -verify-machineinstrs -O0 \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,FALLBACK,VERIFY,VERIFY-O0

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   --debugify-and-strip-all-safe=0 \
; RUN:   -verify-machineinstrs=0 -O0 -aarch64-enable-global-isel-at-O=0 -global-isel-abort=1 \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,NOFALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   --debugify-and-strip-all-safe=0 \
; RUN:   -verify-machineinstrs=0 -O0 -aarch64-enable-global-isel-at-O=0 -global-isel-abort=2  \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,FALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   --debugify-and-strip-all-safe=0 \
; RUN:   -verify-machineinstrs=0 -global-isel \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix NOFALLBACK --check-prefix ENABLED-O1

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   --debugify-and-strip-all-safe=0 \
; RUN:   -verify-machineinstrs=0 -global-isel -global-isel-abort=2 \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix FALLBACK  --check-prefix ENABLED-O1

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   --debugify-and-strip-all-safe=0 \
; RUN:   -verify-machineinstrs=0 -O1 -aarch64-enable-global-isel-at-O=3 \
; RUN:   | FileCheck %s --check-prefix ENABLED  --check-prefix ENABLED-O1

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   --debugify-and-strip-all-safe=0 \
; RUN:   -verify-machineinstrs=0 -O1 -aarch64-enable-global-isel-at-O=0 \
; RUN:   | FileCheck %s --check-prefix DISABLED

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   --debugify-and-strip-all-safe=0 \
; RUN:   -verify-machineinstrs=0 -aarch64-enable-global-isel-at-O=-1 \
; RUN:   | FileCheck %s --check-prefix DISABLED

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   --debugify-and-strip-all-safe=0 \
; RUN:   -verify-machineinstrs=0 | FileCheck %s --check-prefix DISABLED

; RUN: llc -mtriple=aarch64-- -fast-isel=0 -global-isel=false \
; RUN:   --debugify-and-strip-all-safe=0 \
; RUN:   -debug-pass=Structure %s -o /dev/null 2>&1 -verify-machineinstrs=0 \
; RUN:   | FileCheck %s --check-prefix DISABLED

; ENABLED:       IRTranslator
; VERIFY-NEXT:   Verify generated machine code
; ENABLED-NEXT:  Analysis for ComputingKnownBits
; ENABLED-O1-NEXT:  MachineDominator Tree Construction
; ENABLED-NEXT:  PreLegalizerCombiner
; VERIFY-NEXT:   Verify generated machine code
; ENABLED-NEXT:  Analysis containing CSE Info
; ENABLED-NEXT:  Legalizer
; VERIFY-NEXT:   Verify generated machine code
; ENABLED:  RegBankSelect
; VERIFY-NEXT:   Verify generated machine code
; ENABLED-NEXT:  Localizer
; VERIFY-O0-NEXT:   Verify generated machine code
; ENABLED-NEXT: Analysis for ComputingKnownBits
; ENABLED-NEXT:  InstructionSelect
; ENABLED-O1-NEXT:  AArch64 Post Select Optimizer
; VERIFY-NEXT:   Verify generated machine code
; ENABLED-NEXT:  ResetMachineFunction

; FALLBACK:       AArch64 Instruction Selection
; NOFALLBACK-NOT: AArch64 Instruction Selection

; DISABLED-NOT: IRTranslator

; DISABLED: AArch64 Instruction Selection
; DISABLED: Finalize ISel and expand pseudo-instructions

define void @empty() {
  ret void
}
