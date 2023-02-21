; Test to ensure that if a definition is imported, already-present declarations
; are updated as necessary: Definitions from the same module may be optimized
; together. Thus care must be taken when importing only a subset of the
; definitions from a module (because other referenced definitions from that
; module may have been changed by the optimizer and may no longer match
; declarations already present in the module being imported into).

; Generate bitcode and index, and run the function import.
; `Inputs/attr_fixup_dae_noundef.ll` contains the post-"Dead Argument Elimination" IR, which
; removed the `noundef` from `@inner`.
; RUN: opt -module-summary %p/Inputs/attr_fixup_dae_noundef.ll -o %t.inputs.attr_fixup_dae_noundef.bc
; RUN: opt -module-summary %s -o %t.main.bc
; RUN: llvm-lto -thinlto -o %t.summary %t.main.bc %t.inputs.attr_fixup_dae_noundef.bc
; RUN: opt -passes=function-import -summary-file %t.summary.thinlto.bc %t.main.bc -S 2>&1 \
; RUN:   | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @main()  {
  call void @outer(i32 noundef 1)
  call void @inner(i32 noundef 1)
  ret void
}

; Because `@inner` is `noinline`, it should not get imported. However, the
; `noundef` should be removed.
; CHECK: declare void @inner(i32)
declare void @inner(i32 noundef)

; `@outer` should get imported.
; CHECK: define available_externally void @outer(i32 noundef %arg)
; CHECK-NEXT: call void @inner(i32 poison)
declare void @outer(i32 noundef)
