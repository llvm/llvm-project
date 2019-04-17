; RUN: opt < %s -basicaa -argpromotion -mem2reg -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

define internal i32 @test(i32* %X, i32* %Y) {
; CHECK-LABEL: define internal i32 @test(i32 %X.val, i32 %Y.val)
  %A = load i32, i32* %X
  %B = load i32, i32* %Y
  %C = add i32 %A, %B
  ret i32 %C
}

define internal i32 @caller(i32* %B) {
; CHECK-LABEL: define internal i32 @caller(i32 %B.val1)
  %A = alloca i32
  store i32 1, i32* %A
  %C = call i32 @test(i32* %A, i32* %B)
; CHECK: call i32 @test(i32 1, i32 %B.val1)
  ret i32 %C
}

define i32 @callercaller() {
; CHECK-LABEL: define i32 @callercaller()
  %B = alloca i32
  store i32 2, i32* %B
  %X = call i32 @caller(i32* %B)
; CHECK: call i32 @caller(i32 2)
  ret i32 %X
}

