; Test that the strpbrk folder doesn't simplify a call to the function
; declared with an incompatible prototype.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [12 x i8] c"hello world\00"
@w = constant [2 x i8] c"w\00"

declare i8 @strpbrk(ptr, ptr)

; Check that 'strpbrk' functions with the wrong prototype aren't simplified.

define i8 @test_no_simplify1() {
; CHECK-LABEL: @test_no_simplify1(

  %ret = call i8 @strpbrk(ptr @hello, ptr @w)
; CHECK-NEXT: %ret = call i8 @strpbrk
  ret i8 %ret
; CHECK-NEXT: ret i8 %ret
}
