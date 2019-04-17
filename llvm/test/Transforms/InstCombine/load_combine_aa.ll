; RUN: opt -basicaa -instcombine -S < %s | FileCheck %s

; CHECK-LABEL: @test_load_combine_aa(
; CHECK: %[[V:.*]] = load i32, i32* %0
; CHECK: store i32 0, i32* %3
; CHECK: store i32 %[[V]], i32* %1
; CHECK: store i32 %[[V]], i32* %2
define void @test_load_combine_aa(i32*, i32*, i32*, i32* noalias) {
  %a = load i32, i32* %0
  store i32 0, i32* %3
  %b = load i32, i32* %0
  store i32 %a, i32* %1
  store i32 %b, i32* %2
  ret void
}
