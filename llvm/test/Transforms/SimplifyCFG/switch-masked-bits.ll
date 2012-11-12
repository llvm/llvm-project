; RUN: opt -S -simplifycfg < %s | FileCheck %s

define i32 @test1(i32 %x) nounwind {
  %i = shl i32 %x, 1
  switch i32 %i, label %a [
    i32 21, label %b
    i32 24, label %c
  ]

a:
  ret i32 0
b:
  ret i32 3
c:
  ret i32 5
; CHECK: @test1
; CHECK: %cond = icmp eq i32 %i, 24
; CHECK: %. = select i1 %cond, i32 5, i32 0
; CHECK: ret i32 %.
}


define i32 @test2(i32 %x) nounwind {
  %i = shl i32 %x, 1
  switch i32 %i, label %a [
    i32 21, label %b
    i32 23, label %c
  ]

a:
  ret i32 0
b:
  ret i32 3
c:
  ret i32 5
; CHECK: @test2
; CHECK: ret i32 0
}
