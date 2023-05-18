; RUN: opt < %s -S -passes=globalopt | FileCheck %s

@zero = internal global [10 x i32] zeroinitializer

define i32 @test1(i64 %idx) nounwind {
  %arrayidx = getelementptr inbounds [10 x i32], ptr @zero, i64 0, i64 %idx
  %l = load i32, ptr %arrayidx
  ret i32 %l
; CHECK-LABEL: @test1(
; CHECK: ret i32 0
}
