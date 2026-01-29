;; Make sure that we always emit an assembly syntax directive for X86.
; RUN: llc < %s -mtriple=x86_64 | FileCheck %s --check-prefix=ATT
; RUN: llc < %s -mtriple=x86_64 -x86-asm-syntax=att | FileCheck %s --check-prefix=ATT
; RUN: llc < %s -mtriple=x86_64 -x86-asm-syntax=intel | FileCheck %s --check-prefix=INTEL

; INTEL: .intel_syntax noprefix
; ATT: .att_syntax
