; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that nested zero-size arrays are legalized to a pointer.

@nested_zero_array = global [2 x [0 x i32]] zeroinitializer

; CHECK: @nested_zero_array = global ptr addrspace(4) null
