; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; Instcombine should be able to do trivial CSE of loads.

define i32 @test1(i32* %p) {
  %t0 = getelementptr i32* %p, i32 1
  %y = load i32* %t0
  %t1 = getelementptr i32* %p, i32 1
  %x = load i32* %t1
  %a = sub i32 %y, %x
  ret i32 %a
; CHECK: @test1
; CHECK: ret i32 0
}


; PR7429
@.str = private constant [4 x i8] c"XYZ\00"
define float @test2() {
  %tmp = load float* bitcast ([4 x i8]* @.str to float*), align 1
  ret float %tmp
  
; CHECK: @test2
; CHECK: ret float 0x3806965600000000
}