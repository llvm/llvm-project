; RUN: opt < %s -passes=gvn-hoist -S | FileCheck %s

define void @func() {
; CHECK-LABEL: @func()
; CHECK:       bb6:
; CHECK:         store i64 0, ptr undef, align 8
; CHECK:       bb7:
; CHECK-NOT:     store i64 0, ptr undef, align 8
; CHECK:       bb8:
; CHECK-NOT:     store i64 0, ptr undef, align 8

entry:
  br label %bb1

bb1:
  br label %bb2

bb2:
  br label %bb3

bb3:
  br i1 undef, label %bb4, label %bb2

bb4:
  br i1 undef, label %bb5, label %bb3

bb5:
  br label %bb6

bb6:
  br i1 undef, label %bb7, label %bb8

bb7:
  store i64 0, ptr undef, align 8
  unreachable

bb8:
  store i64 0, ptr undef, align 8
  ret void
}
