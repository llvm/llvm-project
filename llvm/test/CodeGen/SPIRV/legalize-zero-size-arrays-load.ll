; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that load of zero-size array is replaced with poison and removed.

define void @test_load_zero_array(ptr %ptr) {
; CHECK-LABEL: @test_load_zero_array(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret void
entry:
  %val = load [0 x i32], ptr %ptr, align 4
  ret void
}
