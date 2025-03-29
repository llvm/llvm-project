
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#INT64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PTRINT8:]] = OpTypePointer Workgroup %[[#INT8]]
; CHECK-DAG: %[[#CONST:]] = OpConstant %[[#INT64]] 1

; CHECK: %[[#PARAM1:]] = OpFunctionParameter %[[#PTRINT8]]
define spir_kernel void @test1(ptr addrspace(3) %address) {
; CHECK: %[[#]] = OpInBoundsPtrAccessChain %[[#PTRINT8]] %[[#PARAM1]] %[[#CONST]]
  %cast = bitcast ptr addrspace(3) %address to ptr addrspace(3)
  %gep = getelementptr inbounds i8, ptr addrspace(3) %cast, i64 1
  ret void
}

; CHECK: %[[#PARAM2:]] = OpFunctionParameter %[[#PTRINT8]]
define spir_kernel void @test2(ptr addrspace(3) %address) {
; CHECK: %[[#]] = OpInBoundsPtrAccessChain %[[#PTRINT8]] %[[#PARAM2]] %[[#CONST]]
  %gep = getelementptr inbounds i8, ptr addrspace(3) %address, i64 1
  ret void
}
