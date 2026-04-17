; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that an intrinsic global variable with appending linkage is properly legalized.

@llvm.global_ctors = appending addrspace(1) global [0 x { i32, ptr, ptr }] zeroinitializer

; CHECK: @llvm.global_ctors = extern_weak addrspace(1) global ptr addrspace(4)
