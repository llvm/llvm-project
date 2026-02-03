; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_INTEL_variable_length_array %s -o - -filetype=obj | spirv-val %}

; Test that zero-size array alloca with dynamic count allocates element type with count.

define void @test_alloca_with_count(i32 %n) {
; CHECK-LABEL: @test_alloca_with_count(
; CHECK-NEXT:    [[ARR:%.*]] = alloca ptr addrspace(4), align 4
; CHECK-NEXT:    ret void
  %arr = alloca [0 x i32], i32 %n, align 4
  ret void
}
