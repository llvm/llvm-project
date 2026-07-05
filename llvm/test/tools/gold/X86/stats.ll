; REQUIRES: asserts

; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext  -shared \
; RUN:    -m elf_x86_64 \
; RUN:    -plugin-opt=-stats %t.o -o %t2 2>&1 | FileCheck %s

; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext  -shared \
; RUN:    -m elf_x86_64 \
; RUN:    -plugin-opt=thinlto \
; RUN:    -plugin-opt=thinlto-index-only \
; RUN:    -plugin-opt=-stats %t.o -o %t2 2>&1 | FileCheck %s

; CHECK: Statistics Collected

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
