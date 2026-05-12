; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#I32]]
; CHECK-DAG: %[[#VPTR2:]] = OpTypeVector %[[#PTR]] 2
; CHECK-DAG: %[[#VPTR4:]] = OpTypeVector %[[#PTR]] 4

; The <1 x ptr> GEP collapses to a single scalar OpPtrAccessChain; no
; vector-of-pointers value is materialized.
; CHECK: OpFunction
; CHECK: OpPtrAccessChain %[[#PTR]]
; CHECK-NOT: OpCompositeInsert
; CHECK: OpFunctionEnd
define spir_kernel void @test_vector_gep_v1(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <1 x i64> <i64 5>
  %elem = extractelement <1 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK: OpFunction
; CHECK: OpPtrAccessChain %[[#PTR]]
; CHECK: OpCompositeInsert %[[#VPTR2]]
; CHECK: OpPtrAccessChain %[[#PTR]]
; CHECK: OpCompositeInsert %[[#VPTR2]]
; CHECK: OpFunctionEnd
define spir_kernel void @test_vector_gep_v2(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <2 x i64> zeroinitializer
  %elem = extractelement <2 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK: OpFunction
; CHECK: OpPtrAccessChain %[[#PTR]]
; CHECK: OpCompositeInsert %[[#VPTR4]]
; CHECK: OpPtrAccessChain %[[#PTR]]
; CHECK: OpCompositeInsert %[[#VPTR4]]
; CHECK: OpPtrAccessChain %[[#PTR]]
; CHECK: OpCompositeInsert %[[#VPTR4]]
; CHECK: OpPtrAccessChain %[[#PTR]]
; CHECK: OpCompositeInsert %[[#VPTR4]]
; CHECK: OpFunctionEnd
define spir_kernel void @test_vector_gep_v4(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <4 x i64> zeroinitializer
  %elem = extractelement <4 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}
