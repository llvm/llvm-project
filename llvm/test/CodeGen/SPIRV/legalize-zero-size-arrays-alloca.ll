; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that alloca of zero-size array allocates element type instead.

define void @test_alloca_zero_array() {
; CHECK-LABEL: @test_alloca_zero_array(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARR:%.*]] = alloca ptr addrspace(4), align 4
; CHECK-NEXT:    ret void
entry:
  %arr = alloca [0 x i32], align 4
  ret void
}
