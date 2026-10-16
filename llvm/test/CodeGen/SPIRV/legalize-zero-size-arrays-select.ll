; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s

; Test that select of zero-size array is replaced with poison.

; Can't run spirv-val as function signatures are not handled.
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define [0 x i32] @test_select_zero_array(i1 %cond, [0 x i32] %a) {
; CHECK-LABEL: @test_select_zero_array(
; CHECK-NEXT:    ret [0 x i32] poison
  %result = select i1 %cond, [0 x i32] zeroinitializer, [0 x i32] %a
  ret [0 x i32] %result
}
