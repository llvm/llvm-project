; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/llvm.used.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc


; RUN: llvm-lto -thinlto-action=import %t2.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s
; CHECK: define available_externally void @globalfunc


target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"



define internal void @_ZN12_GLOBAL__N_16Module4dumpEv() {
    ret void
}
@llvm.used = appending global [1 x ptr] [ptr @_ZN12_GLOBAL__N_16Module4dumpEv], section "llvm.metadata"


define void @globalfunc() #0 {
entry:
  ret void
}
