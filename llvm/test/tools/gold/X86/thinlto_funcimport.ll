; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/thinlto_funcimport.ll -o %t2.bc

; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=save-temps \
; RUN:    --plugin-opt=thinlto \
; RUN:    -shared %t1.bc %t2.bc -o %t
; RUN: llvm-dis %t2.bc.3.import.bc -o - | FileCheck %s
; CHECK: define available_externally void @foo()

; We shouldn't do any importing at -O0
; rm -f %t2.bc.3.import.bc
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=save-temps \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=O0 \
; RUN:    -shared %t1.bc %t2.bc -o %t
; RUN: llvm-dis %t2.bc.3.import.bc -o - | FileCheck %s --check-prefix=CHECKO0
; CHECKO0: declare void @foo(...)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() #0 {
entry:
  ret void
}
