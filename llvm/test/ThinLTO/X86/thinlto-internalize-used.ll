; RUN: opt -module-summary -o %t.bc %s
; RUN: opt -module-summary -o %t-main.bc %S/Inputs/thinlto-internalize-used2.ll
; RUN: llvm-lto -thinlto-action=thinlink %t.bc %t-main.bc -o %t-index.bc
; RUN: llvm-lto -thinlto-action=internalize -thinlto-index %t-index.bc %t.bc -o %t.promote.bc
; RUN: llvm-dis %t.promote.bc -o - | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@llvm.used = appending global [1 x ptr] [ptr @foo], section "llvm.metadata"

; Make sure foo is not internalized.
; CHECK: define i32 @foo()
define i32 @foo() {
  ret i32 0
}

define hidden i32 @bar() {
  ret i32 0
}

