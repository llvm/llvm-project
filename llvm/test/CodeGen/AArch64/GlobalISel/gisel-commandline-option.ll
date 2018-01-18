; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -O0 | FileCheck %s --check-prefix ENABLED --check-prefix ENABLED-O0 --check-prefix FALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -O0 -aarch64-enable-global-isel-at-O=0 -global-isel-abort=1 \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix ENABLED-O0 --check-prefix NOFALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -O0 -aarch64-enable-global-isel-at-O=0 -global-isel-abort=2  \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix ENABLED-O0 --check-prefix FALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -global-isel \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix NOFALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -global-isel -global-isel-abort=2 \
; RUN:   | FileCheck %s --check-prefix ENABLED --check-prefix FALLBACK

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -O1 -aarch64-enable-global-isel-at-O=3 \
; RUN:   | FileCheck %s --check-prefix ENABLED

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -O1 -aarch64-enable-global-isel-at-O=0 \
; RUN:   | FileCheck %s --check-prefix DISABLED

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -aarch64-enable-global-isel-at-O=-1 \
; RUN:   | FileCheck %s --check-prefix DISABLED

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix DISABLED

; RUN: llc -mtriple=aarch64-- -fast-isel=0 -global-isel=false \
; RUN: -debug-pass=Structure %s -o /dev/null 2>&1 | FileCheck %s --check-prefix DISABLED

; ENABLED:       IRTranslator
; ENABLED-NEXT:  Legalizer
; ENABLED-NEXT:  RegBankSelect
; ENABLED-O0-NEXT:  Localizer
; ENABLED-NEXT:  InstructionSelect
; ENABLED-NEXT:  ResetMachineFunction

; FALLBACK:       AArch64 Instruction Selection
; NOFALLBACK-NOT: AArch64 Instruction Selection

; DISABLED-NOT: IRTranslator

; DISABLED: AArch64 Instruction Selection
; DISABLED: Expand ISel Pseudo-instructions

define void @empty() {
  ret void
}
