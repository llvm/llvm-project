; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that a global variable with zero-size array is transformed to ptr type.

@global_zero_array = global [0 x i32] zeroinitializer

; CHECK: @global_zero_array = global ptr addrspace(4) null
