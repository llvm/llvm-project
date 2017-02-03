; RUN: llc %s -o - -mtriple=thumbv8m.base | FileCheck %s

define void @test() {
; CHECK-LABEL: test:
entry:
  %call = tail call i32 @foo()
  %tail = tail call i32 @foo()
  ret void
; CHECK: bl foo
; CHECK: bl foo
; CHECK-NOT: b foo
}

define void @test2() {
; CHECK-LABEL: test2:
entry:
  %tail = tail call i32 @foo()
  ret void
; CHECK: b foo
; CHECK-NOT: bl foo
}

declare i32 @foo()
