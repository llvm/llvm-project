; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as -o %t/bcsection.bc %s

; RUN: llvm-mc -I=%t -filetype=obj -triple=x86_64-unknown-unknown -o %t/bcsection.bco %p/Inputs/bcsection.s
; RUN: llvm-nm -no-llvm-bc %t/bcsection.bco | count 0
; RUN: %gold -r -o %t/bcsection.o -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext %t/bcsection.bco
; RUN: llvm-nm -no-llvm-bc %t/bcsection.o | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; CHECK: main
define i32 @main() {
  ret i32 0
}
