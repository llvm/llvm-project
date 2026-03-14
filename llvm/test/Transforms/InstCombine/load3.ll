; RUN: opt < %s -passes=instcombine -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.0.0"

; Instcombine should be able to do trivial CSE of loads.

define i32 @test1(ptr %p) {
  %t0 = getelementptr i32, ptr %p, i32 1
  %y = load i32, ptr %t0
  %t1 = getelementptr i32, ptr %p, i32 1
  %x = load i32, ptr %t1
  %a = sub i32 %y, %x
  ret i32 %a
; CHECK-LABEL: @test1(
; CHECK: ret i32 0
}


; PR7429
@.str = private constant [4 x i8] c"XYZ\00"
define float @test2() {
  %tmp = load float, ptr @.str, align 1
  ret float %tmp
  
; CHECK-LABEL: @test2(
; CHECK: ret float 0x3806965600000000
}

@rslts32 = global [36 x i32] zeroinitializer, align 4

@expect32 = internal constant [36 x i32][ i32 1, i32 2, i32 0, i32 100, i32 3,
i32 4, i32 0, i32 -7, i32 4, i32 4, i32 8, i32 8, i32 1, i32 3, i32 8, i32 3,
i32 4, i32 -2, i32 2, i32 8, i32 83, i32 77, i32 8, i32 17, i32 77, i32 88, i32
22, i32 33, i32 44, i32 88, i32 77, i32 4, i32 4, i32 7, i32 -7, i32 -8] ,
align 4

; PR14986
define void @test3() nounwind {
; This is a weird way of computing zero.
  %l = load i32, ptr getelementptr ([36 x i32], ptr @expect32, i32 29826161, i32 28), align 4
  store i32 %l, ptr getelementptr ([36 x i32], ptr @rslts32, i32 29826161, i32 28), align 4
  ret void

; CHECK-LABEL: @test3(
; CHECK: store i32 1, ptr @rslts32
}
