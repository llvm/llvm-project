; RUN: opt -instcombine -S < %s | FileCheck %s

; CHECK-LABEL: define i32 @t(
; CHECK: ret i32 0

define i32 @t(i32* %p0, i32* %p1) {
entry:
  store i32 0, i32* %p1
  store i32 0, i32* %p0
  %v = load i32, i32* %p1
  ret i32 %v
}
