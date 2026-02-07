; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that struct with zero-size array field becomes pointer.

%struct.with_zero = type { i32, [0 x i32], i32 }

@global_struct = global %struct.with_zero zeroinitializer

; CHECK: %struct.with_zero.legalized = type { i32, ptr addrspace(4), i32 }
; CHECK: @global_struct = global %struct.with_zero.legalized zeroinitializer
