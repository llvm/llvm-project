; RUN: opt -S -early-cse < %s | FileCheck %s

declare void @readnone_may_unwind() readnone

define void @f(i32* %ptr) {
; CHECK-LABEL: @f(
; CHECK: store i32 100, i32* %ptr
; CHECK: call void @readnone_may_unwind()
; CHECK: store i32 200, i32* %ptr

  store i32 100, i32* %ptr
  call void @readnone_may_unwind()
  store i32 200, i32* %ptr
  ret void
}
