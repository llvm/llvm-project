; Check that we correctly handle the case where we have inline asm and the
; target is not registered. In this case we shouldn't emit an irsymtab.

; RUN: llvm-as -o %t %s
; RUN: llvm-bcanalyzer -dump %t | FileCheck --check-prefix=AS %s

; AS-NOT: <SYMTAB_BLOCK

; RUN: opt -o %t2 %s
; RUN: llvm-bcanalyzer -dump %t2 | FileCheck --check-prefix=OPT %s

; OPT: <SYMTAB_BLOCK

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm "ret"
