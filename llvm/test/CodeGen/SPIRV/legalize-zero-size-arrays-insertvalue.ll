; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s

; Test that insertvalue of zero-size array is removed

%struct.with_zero = type { i32, [0 x i32], i64 }

define void @test_insertvalue_zero_array(ptr %ptr, %struct.with_zero %s) {
; CHECK-LABEL: @test_insertvalue_zero_array(
; CHECK-NEXT:    [[AGG:%.*]] = insertvalue %struct.with_zero %s, i32 42, 0
; CHECK-NEXT:    store %struct.with_zero [[AGG]], ptr %ptr
; CHECK-NEXT:    ret void
  %agg = insertvalue %struct.with_zero %s, i32 42, 0
  %result = insertvalue %struct.with_zero %agg, [0 x i32] zeroinitializer, 1
  store %struct.with_zero %result, ptr %ptr
  ret void
}
