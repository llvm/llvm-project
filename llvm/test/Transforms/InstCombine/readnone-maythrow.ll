; RUN: opt -S -instcombine < %s | FileCheck %s

declare void @readnone_but_may_throw() readnone

define void @f_0(i32* %ptr) {
; CHECK-LABEL: @f_0(
entry:
; CHECK:  store i32 10, i32* %ptr
; CHECK-NEXT:  call void @readnone_but_may_throw()
; CHECK-NEXT:  store i32 20, i32* %ptr, align 4
; CHECK:  ret void

  store i32 10, i32* %ptr
  call void @readnone_but_may_throw()
  store i32 20, i32* %ptr
  ret void
}

define void @f_1(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @f_1(
; CHECK:  store i32 10, i32* %ptr
; CHECK-NEXT:  call void @readnone_but_may_throw()

  store i32 10, i32* %ptr
  call void @readnone_but_may_throw()
  br i1 %cond, label %left, label %merge

left:
  store i32 20, i32* %ptr
  br label %merge

merge:
  ret void
}
