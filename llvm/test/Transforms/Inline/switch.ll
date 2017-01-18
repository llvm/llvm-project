; RUN: opt < %s -inline -inline-threshold=20 -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -inline-threshold=20 -S | FileCheck %s

define i32 @callee(i32 %a) {
  switch i32 %a, label %sw.default [
    i32 0, label %sw.bb0
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
    i32 4, label %sw.bb4
    i32 5, label %sw.bb5
    i32 6, label %sw.bb6
    i32 7, label %sw.bb7
    i32 8, label %sw.bb8
    i32 9, label %sw.bb9
  ]

sw.default:
  br label %return

sw.bb0:
  br label %return

sw.bb1:
  br label %return

sw.bb2:
  br label %return

sw.bb3:
  br label %return

sw.bb4:
  br label %return

sw.bb5:
  br label %return

sw.bb6:
  br label %return

sw.bb7:
  br label %return

sw.bb8:
  br label %return

sw.bb9:
  br label %return

return:
  ret i32 42
}

define i32 @caller(i32 %a) {
; CHECK-LABEL: @caller(
; CHECK: call i32 @callee(

  %result = call i32 @callee(i32 %a)
  ret i32 %result
}
