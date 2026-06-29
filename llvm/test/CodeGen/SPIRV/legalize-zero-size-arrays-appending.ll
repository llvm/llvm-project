; RUN: opt -S -passes=spirv-legalize-zero-size-arrays -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that intrinsic global variables with appending linkage are properly legalized.

@llvm.global_ctors = appending addrspace(1) global [0 x { i32, ptr, ptr }] zeroinitializer
@llvm.global_dtors = appending addrspace(1) global [0 x { i32, ptr, ptr }] zeroinitializer
@llvm.used = external addrspace(1) global [0 x ptr addrspace(1)]
@llvm.compiler.used = external addrspace(1) global [0 x ptr addrspace(1)]

; CHECK: @llvm.global_ctors = extern_weak addrspace(1) global ptr addrspace(4)
; CHECK: @llvm.global_dtors = extern_weak addrspace(1) global ptr addrspace(4)
; CHECK: @llvm.used = external addrspace(1) global ptr addrspace(4)
; CHECK: @llvm.compiler.used = external addrspace(1) global ptr addrspace(4)
