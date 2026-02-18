; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that store of zero-size array is removed.

define void @test_store_zero_array(ptr %ptr) {
; CHECK-LABEL: @test_store_zero_array(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret void
entry:
  store [0 x i32] zeroinitializer, ptr %ptr, align 4
  ret void
}
