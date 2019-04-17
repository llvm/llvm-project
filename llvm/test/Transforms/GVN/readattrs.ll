; RUN: opt -gvn -S -o - < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

declare void @use(i8* readonly nocapture)

define i8 @test() {
  %a = alloca i8
  store i8 1, i8* %a
  call void @use(i8* %a)
  %b = load i8, i8* %a
  ret i8 %b
; CHECK-LABEL: define i8 @test(
; CHECK: call void @use(i8* %a)
; CHECK-NEXT: ret i8 1
}
