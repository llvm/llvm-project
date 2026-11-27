; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that struct with zero-size array field becomes pointer.

%struct.with_zero = type { i32, [0 x i32], i32 }

@global_struct = addrspace(1) global %struct.with_zero zeroinitializer

%struct.mixed = type { [4 x i32], [0 x i32] }

@global_mixed = addrspace(1) global %struct.mixed zeroinitializer

; CHECK: %struct.with_zero.legalized = type { i32, ptr addrspace(4), i32 }
; CHECK: %struct.mixed.legalized = type { [4 x i32], ptr addrspace(4) }
; CHECK: @global_struct = addrspace(1) global %struct.with_zero.legalized zeroinitializer
; CHECK: @global_mixed = addrspace(1) global %struct.mixed.legalized zeroinitializer
