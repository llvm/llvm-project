; RUN: opt -passes=instcombine -S < %s | FileCheck %s

target datalayout = "e-m:e-p:64:64:64-i64:64-f80:128-n8:16:32:64-S128"

; Check that nonnull metadata is propagated from dominating load.
; CHECK-LABEL: @combine_metadata_dominance1(
; CHECK-LABEL: bb1:
; CHECK: load i32*, i32** %p, align 8, !nonnull !0
; CHECK-NOT: load i32*, i32** %p
define void @combine_metadata_dominance1(i32** %p) {
entry:
  %a = load i32*, i32** %p, !nonnull !0
  br label %bb1

bb1:
  %b = load i32*, i32** %p
  store i32 0, i32* %a
  store i32 0, i32* %b
  ret void
}

declare i32 @use(i32*, i32) readonly

; Check that nonnull from the dominated load does not get propagated.
; There are some cases where it would be safe to keep it.
; CHECK-LABEL: @combine_metadata_dominance2(
; CHECK-NOT: nonnull
define void @combine_metadata_dominance2(i32** %p, i1 %c1) {
entry:
  %a = load i32*, i32** %p
  br i1 %c1, label %bb1, label %bb2

bb1:
  %b = load i32*, i32** %p, !nonnull !0
  store i32 0, i32* %a
  store i32 0, i32* %b
  ret void

bb2:
  ret void
}


!0 = !{}
