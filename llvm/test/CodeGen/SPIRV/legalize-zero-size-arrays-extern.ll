; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that an externally initialized global with no initializer

@global_zero_array = external global [0 x i32]

; CHECK: @global_zero_array = external global ptr addrspace(4)
