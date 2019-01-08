; REQUIRES: asserts

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -O0 -global-isel=false -debug-only=isel \
; RUN:   | FileCheck %s --check-prefixes=DISABLED,FASTISEL

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -O1 -global-isel=false -debug-only=isel \
; RUN:   | FileCheck %s --check-prefixes=DISABLED,NOFASTISEL

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -O0 -fast-isel=false -global-isel=false \
; RUN:   -debug-only=isel \
; RUN:   | FileCheck %s --check-prefixes=DISABLED,NOFASTISEL

; RUN: llc -mtriple=aarch64-- -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:   -verify-machineinstrs=0 -O1 -fast-isel=false -global-isel=false \
; RUN:   -debug-only=isel \
; RUN:   | FileCheck %s --check-prefixes=DISABLED,NOFASTISEL

; Check that the right instruction selector is chosen when using
; -global-isel=false. FastISel should be used at -O0 (unless -fast-isel=false is
; also present) and SelectionDAG otherwise.

; DISABLED-NOT: IRTranslator

; DISABLED: AArch64 Instruction Selection
; DISABLED: Expand ISel Pseudo-instructions

; FASTISEL: Enabling fast-isel
; NOFASTISEL-NOT: Enabling fast-isel

define void @empty() {
  ret void
}
