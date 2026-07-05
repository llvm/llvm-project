; Check that llvm-nm doesn't crash on COFF-specific assembly directives
; (PR36789).

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-nm %t.bc

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24210"

module asm ".text"
module asm ".def foo; .scl 2; .type 32; .endef"
module asm "foo:"
module asm "  ret"
