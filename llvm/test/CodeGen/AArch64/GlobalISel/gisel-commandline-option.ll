; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -O0 \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,ENABLED-O0,FALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs -O0 \
; RUN:   | FileCheck %s --check-prefixes=ENABLED,ENABLED-O0,FALLBACK,VERIFY,VERIFY-O0

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -O0 -aarch64-enable-global-isel-at-O=0 -global-isel-abort=1 \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix ENABLED-O0 --check-prefix NOFALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -O0 -aarch64-enable-global-isel-at-O=0 -global-isel-abort=2  \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix ENABLED-O0 --check-prefix FALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -global-isel \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix NOFALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -global-isel -global-isel-abort=2 \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix FALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -O1 -aarch64-enable-global-isel-at-O=3 \
; RUN:   | FileCheck %s --check-prefix ENABLED

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -O1 -aarch64-enable-global-isel-at-O=0 \
; RUN:   | FileCheck %s --check-prefix DISABLED

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -aarch64-enable-global-isel-at-O=-1 \
; RUN:   | FileCheck %s --check-prefix DISABLED

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 | FileCheck %s --check-prefix DISABLED

; RUN: llc -mtriple=aarch64-- -fast-isel=0 -global-isel=false \
; RUN:   -debug-pass=Structure %s -o /dev/null 2>&1 -verify-machineinstrs=0 \
; RUN:   | FileCheck %s --check-prefix DISABLED

; ENABLED:       IRTranslator
; VERIFY-NEXT:   Verify generated machine code
; ENABLED-NEXT:  PreLegalizerCombiner
; VERIFY-NEXT:   Verify generated machine code
; ENABLED-NEXT:  Analysis containing CSE Info
; ENABLED-NEXT:  Legalizer
; VERIFY-NEXT:   Verify generated machine code
; ENABLED-NEXT:  RegBankSelect
; VERIFY-NEXT:   Verify generated machine code
; ENABLED-O0-NEXT:  Localizer
; VERIFY-O0-NEXT:   Verify generated machine code
; ENABLED-NEXT:  InstructionSelect
; VERIFY-NEXT:   Verify generated machine code
; ENABLED-NEXT:  ResetMachineFunction

; FALLBACK:       AArch64 Instruction Selection
; NOFALLBACK-NOT: AArch64 Instruction Selection

; DISABLED-NOT: IRTranslator

; DISABLED: AArch64 Instruction Selection
; DISABLED: Expand ISel Pseudo-instructions

define void @empty() {
  ret void
}
