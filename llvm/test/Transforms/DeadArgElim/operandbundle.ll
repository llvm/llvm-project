; RUN: opt < %s -deadargelim -S | FileCheck %s

define internal void @f(i32 %arg) {
entry:
  call void @g() [ "foo"(i32 %arg) ]
  ret void
}

; CHECK-LABEL: define internal void @f(
; CHECK: call void @g() [ "foo"(i32 %arg) ]

declare void @g()
