; RUN: llc < %s -verify-coalescing
; PR10998

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-unknown-freebsd8.2"

define void @test(i32 %arg1) nounwind align 2 {
bb11:
  %tmp13 = and i32 %arg1, 7
  %tmp14 = add i32 %tmp13, -5
  switch i32 %tmp13, label %bb18 [
    i32 0, label %bb21
    i32 4, label %bb22
    i32 3, label %bb21
    i32 2, label %bb19
  ]

bb18:
  %tmp202 = call i32 @f() nounwind
  unreachable

bb19:
  %tmp20 = call i32 @f() nounwind
  br label %bb24

bb21:
  %tmp203 = call i32 @f() nounwind
  br label %bb24

bb22:
  %tmp23 = call i32 @f() nounwind
  br label %bb24

bb24:
  %tmp15 = icmp ult i32 %tmp14, 2
  %tmp55 = select i1 %tmp15, i32 45, i32 44
  %tmp56 = call i32 @f2(i32 %tmp55)
  unreachable
}

declare i32 @f()

declare i32 @f2(i32)
