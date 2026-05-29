; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

target triple = "spirv64-amd-amdhsa"

declare spir_kernel void @arbitrary(ptr addrspace(1))

declare spir_func <4 x i8> @llvm.masked.load.v4i8.p7(ptr addrspace(7) captures(none), i32 immarg, <4 x i1>, <4 x i8>)
; CHECK-NOT: llvm_masked_load
