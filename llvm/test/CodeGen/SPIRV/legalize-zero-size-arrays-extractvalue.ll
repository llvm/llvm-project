; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s

; Test that extractvalue of zero-size array is replaced with poison.

; Can't run spirv-val as function signatures are not handled.
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define [0 x i32] @test_extractvalue_zero_array() {
; CHECK-LABEL: @test_extractvalue_zero_array(
; CHECK-NEXT:    ret [0 x i32] poison
  %arr = extractvalue [1 x [0 x i32]] zeroinitializer, 0
  ret [0 x i32] %arr
}
