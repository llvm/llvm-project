; RUN: opt < %s -tailcallelim -S | FileCheck %s

; Test that we don't tail call in a functions that calls returns_twice
; functions.

declare void @bar()

; CHECK: foo1
; CHECK-NOT: tail call void @bar()

define void @foo1(i32* %x) {
bb:
  %tmp75 = tail call i32 @setjmp(i32* %x)
  call void @bar()
  ret void
}

declare i32 @setjmp(i32*) returns_twice

; CHECK: foo2
; CHECK-NOT: tail call void @bar()

define void @foo2(i32* %x) {
bb:
  %tmp75 = tail call i32 @zed2(i32* %x)
  call void @bar()
  ret void
}
declare i32 @zed2(i32*) returns_twice
