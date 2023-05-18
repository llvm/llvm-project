; Test that the strcmp library call simplifier works correctly.
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"
@hell = constant [5 x i8] c"hell\00"

declare i16 @strcmp(ptr, ptr)

define i16 @test_nosimplify() {
; CHECK-LABEL: @test_nosimplify(
; CHECK: call i16 @strcmp
; CHECK: ret i16 %temp1

  %temp1 = call i16 @strcmp(ptr @hell, ptr @hello)
  ret i16 %temp1
}
