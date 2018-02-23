; RUN: llvm-as -o %T/bcsection.bc %s

; RUN: llvm-mc -I=%T -filetype=obj -triple=x86_64-unknown-unknown -o %T/bcsection.bco %p/Inputs/bcsection.s
; RUN: llvm-nm -no-llvm-bc %T/bcsection.bco | count 0
; RUN: %gold -r -o %T/bcsection.o -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold.so %T/bcsection.bco
; RUN: llvm-nm -no-llvm-bc %T/bcsection.o | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; CHECK: main
define i32 @main() {
  ret i32 0
}
