; RUN: opt -S -passes=instcombine < %s | FileCheck %s

declare void @readnone_but_may_throw() readnone

define void @f_0(ptr %ptr) {
; CHECK-LABEL: @f_0(
entry:
; CHECK:  store i32 10, ptr %ptr
; CHECK-NEXT:  call void @readnone_but_may_throw()
; CHECK-NEXT:  store i32 20, ptr %ptr, align 4
; CHECK:  ret void

  store i32 10, ptr %ptr
  call void @readnone_but_may_throw()
  store i32 20, ptr %ptr
  ret void
}

define void @f_1(i1 %cond, ptr %ptr) {
; CHECK-LABEL: @f_1(
; CHECK:  store i32 10, ptr %ptr
; CHECK-NEXT:  call void @readnone_but_may_throw()

  store i32 10, ptr %ptr
  call void @readnone_but_may_throw()
  br i1 %cond, label %left, label %merge

left:
  store i32 20, ptr %ptr
  br label %merge

merge:
  ret void
}
