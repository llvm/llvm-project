; RUN: opt -S < %s -correlated-propagation | FileCheck %s

; CHECK: @test
define i16 @test(i32 %a, i1 %b) {
entry:
  %c = icmp eq i32 %a, 0
  br i1 %c, label %left, label %right

right:
  %d = trunc i32 %a to i1
  br label %merge

left:
  br i1 %b, label %merge, label %other

other:
  ret i16 23

merge:
  %f = phi i1 [%b, %left], [%d, %right]
; CHECK: select i1 %f, i16 1, i16 0 
  %h = select i1 %f, i16 1, i16 0 
; CHECK: ret i16 %h
  ret i16 %h
}