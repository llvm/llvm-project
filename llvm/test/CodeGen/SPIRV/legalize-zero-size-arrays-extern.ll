; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that an externally initialized global with no initializer.

@global_zero_array = external global [0 x i32]

; CHECK: @global_zero_array = external global ptr addrspace(4)
define spir_kernel void @test_extern(ptr %ptr) {
; CHECK-LABEL: @test_extern(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret void
entry:
  %val = load [0 x i32], ptr @global_zero_array, align 4
  store [0 x i32] %val, ptr %ptr, align 4
  ret void
}