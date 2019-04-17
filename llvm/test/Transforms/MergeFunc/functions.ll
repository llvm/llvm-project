; RUN: opt -S -mergefunc < %s | FileCheck %s

; Be sure we don't merge cross-referenced functions of same type.

; CHECK-LABEL: @left
; CHECK-LABEL: entry-block
; CHECK-LABEL: call void @right(i64 %p)
define void @left(i64 %p) {
entry-block:
  call void @right(i64 %p)
  call void @right(i64 %p)
  call void @right(i64 %p)
  call void @right(i64 %p)
  ret void
}

; CHECK-LABEL: @right
; CHECK-LABEL: entry-block
; CHECK-LABEL: call void @left(i64 %p)
define void @right(i64 %p) {
entry-block:
  call void @left(i64 %p)
  call void @left(i64 %p)
  call void @left(i64 %p)
  call void @left(i64 %p)
  ret void
}
