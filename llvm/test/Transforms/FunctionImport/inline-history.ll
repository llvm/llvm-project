; Test to ensure that we don't import !inline_history metadata on calls since
; doing so may end up importing other function declarations in a way that isn't
; tracked by ThinLTO, breaking IR semantics.

; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/inline-history.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc
; RUN: opt -passes=function-import -summary-file %t3.thinlto.bc %t.bc -S | FileCheck %s --implicit-check-not=inline_history

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller() {
  call void @imported_func()
  ret void
}

declare void @imported_func()

; CHECK: define available_externally void @imported_func()
; CHECK-NEXT: call void @another_func()
