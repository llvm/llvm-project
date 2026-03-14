; RUN: opt < %s -passes=tailcallelim -verify-dom-info -S | FileCheck %s

; Test that we don't tail call in a functions that calls returns_twice
; functions.

declare void @bar()

; CHECK: foo1
; CHECK-NOT: tail call void @bar()

define void @foo1(ptr %x) {
bb:
  %tmp75 = tail call i32 @setjmp(ptr %x)
  call void @bar()
  ret void
}

declare i32 @setjmp(ptr) returns_twice

; CHECK: foo2
; CHECK-NOT: tail call void @bar()

define void @foo2(ptr %x) {
bb:
  %tmp75 = tail call i32 @zed2(ptr %x)
  call void @bar()
  ret void
}
declare i32 @zed2(ptr) returns_twice
