; Test that the strcat libcall simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"
@empty = constant [1 x i8] c"\00"
@a = common global [32 x i8] zeroinitializer, align 1

declare ptr @strcat(ptr, ptr)

define void @test_simplify1() {
; CHECK-LABEL: @test_simplify1(
; CHECK-NOT: call ptr @strcat
; CHECK: ret void

  call ptr @strcat(ptr @a, ptr @hello)
  ret void
}

define void @test_simplify2() {
; CHECK-LABEL: @test_simplify2(
; CHECK-NEXT: ret void

  call ptr @strcat(ptr @a, ptr @empty)
  ret void
}
