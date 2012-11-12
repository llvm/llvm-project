; Test that the strstr library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@null = private constant [1 x i8] zeroinitializer

declare i8 @strstr(i8*, i8*)

define i8 @test_no_simplify1(i8* %str) {
; CHECK: @test_no_simplify1
  %pat = getelementptr inbounds [1 x i8]* @null, i32 0, i32 0
  %ret = call i8 @strstr(i8* %str, i8* %pat)
; CHECK-NEXT: call i8 @strstr
  ret i8 %ret
; CHECK-NEXT: ret i8 %ret
}
