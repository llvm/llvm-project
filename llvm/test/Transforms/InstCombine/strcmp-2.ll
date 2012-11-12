; Test that the strcmp library call simplifier works correctly.
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"
@hell = constant [5 x i8] c"hell\00"

declare i16 @strcmp(i8*, i8*)

define i16 @test_nosimplify() {
; CHECK: @test_nosimplify
; CHECK: call i16 @strcmp
; CHECK: ret i16 %temp1

  %str1 = getelementptr inbounds [5 x i8]* @hell, i32 0, i32 0
  %str2 = getelementptr inbounds [6 x i8]* @hello, i32 0, i32 0
  %temp1 = call i16 @strcmp(i8* %str1, i8* %str2)
  ret i16 %temp1
}
