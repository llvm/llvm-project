;; Test that we respect the assembler dialect when parsing module-level inline asm.
; RUN: not llc < %s -mtriple=x86_64 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: llc < %s -mtriple=x86_64 -x86-asm-syntax=intel | FileCheck %s

; ERR: <inline asm>:1:1: error: unknown use of instruction mnemonic without a size suffix

; CHECK: .intel_syntax noprefix
; CHECK: mov eax, eax

module asm "mov eax, eax"
