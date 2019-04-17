; RUN: opt -S -ipsccp < %s | FileCheck %s

declare void @foo()
declare void @goo()
declare void @patatino()

define void @test1(i32 %t) {
  %choice = icmp eq i32 undef, -1
  switch i1 %choice, label %first [i1 0, label %second
                                   i1 1, label %third]
first:
  call void @foo()
  ret void
second:
  call void @goo()
  ret void
third:
  call void @patatino()
  ret void
}

; CHECK: define void @test1(i32 %t) {
; CHECK-NEXT:   br label %second
; CHECK: second:
; CHECK-NEXT:   call void @goo()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
