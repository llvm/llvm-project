; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; CHECK-LABEL: @test_load_combine_aa(
; CHECK: %[[V:.*]] = load i32, ptr %0
; CHECK: store i32 0, ptr %3
; CHECK: store i32 %[[V]], ptr %1
; CHECK: store i32 %[[V]], ptr %2
define void @test_load_combine_aa(ptr, ptr, ptr, ptr noalias) {
  %a = load i32, ptr %0
  store i32 0, ptr %3
  %b = load i32, ptr %0
  store i32 %a, ptr %1
  store i32 %b, ptr %2
  ret void
}
