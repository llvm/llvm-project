; RUN: opt -passes=instcombine -S < %s | FileCheck %s

target datalayout = "e-m:e-p:64:64:64-i64:64-f80:128-n8:16:32:64-S128"

; Check that nonnull metadata is propagated from dominating load.
; CHECK-LABEL: @combine_metadata_dominance1(
; CHECK-LABEL: bb1:
; CHECK: load ptr, ptr %p, align 8, !nonnull !0
; CHECK-NOT: load ptr, ptr %p
define void @combine_metadata_dominance1(ptr %p) {
entry:
  %a = load ptr, ptr %p, !nonnull !0
  br label %bb1

bb1:
  %b = load ptr, ptr %p
  store i32 0, ptr %a
  store i32 0, ptr %b
  ret void
}

declare i32 @use(ptr, i32) readonly

; Check that nonnull from the dominated load does not get propagated.
; There are some cases where it would be safe to keep it.
; CHECK-LABEL: @combine_metadata_dominance2(
; CHECK-NOT: nonnull
define void @combine_metadata_dominance2(ptr %p, i1 %c1) {
entry:
  %a = load ptr, ptr %p
  br i1 %c1, label %bb1, label %bb2

bb1:
  %b = load ptr, ptr %p, !nonnull !0
  store i32 0, ptr %a
  store i32 0, ptr %b
  ret void

bb2:
  ret void
}


!0 = !{}
