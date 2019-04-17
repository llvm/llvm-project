; RUN: opt < %s -inline -S | FileCheck %s
; rdar://7173846
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

define internal void @foo() nounwind ssp {
entry:
  %A = alloca [100 x i32]
  %B = alloca [100 x i32]
  call void @bar([100 x i32]* %A, [100 x i32]* %B) nounwind
  ret void
}

declare void @bar([100 x i32]*, [100 x i32]*)

define void @test() nounwind ssp {
entry:
; CHECK: @test()
; CHECK-NEXT: entry:
; CHECK-NEXT: %A.i = alloca
; CHECK-NEXT: %B.i = alloca
; CHECK-NOT: alloca
  call void @foo() nounwind
  call void @foo() nounwind
  ret void
}
