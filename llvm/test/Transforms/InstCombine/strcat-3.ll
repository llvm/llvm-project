; Test that the strcat folder avoids simplifying a call to the function
; declared with an incompatible type.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"
@empty = constant [1 x i8] c"\00"
@a = common global [32 x i8] zeroinitializer, align 1

; Expected type: ptr @strcat(ptr, ptr).
declare i16 @strcat(ptr, ptr)

define void @test_nosimplify1() {
; CHECK-LABEL: @test_nosimplify1(
; CHECK: call i16 @strcat
; CHECK: ret void

  call i16 @strcat(ptr @a, ptr @hello)
  ret void
}
