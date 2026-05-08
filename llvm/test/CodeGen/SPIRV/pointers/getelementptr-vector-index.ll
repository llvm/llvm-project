; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#INT32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR_INT32:]] = OpTypePointer CrossWorkgroup %[[#INT32]]
; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PTR_INT8:]] = OpTypePointer CrossWorkgroup %[[#INT8]]
; CHECK-DAG: %[[#INT64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#CONST_0:]] = OpConstantNull %[[#INT64]]
; CHECK-LABEL: Begin function test_vector_gep_with_load
; CHECK: %[[#BC1:]] = OpBitcast %[[#PTR_INT8]] %[[#]]
; CHECK: %[[#GEP:]] = OpPtrAccessChain %[[#PTR_INT8]] %[[#BC1]] %[[#CONST_0]]
; CHECK: %[[#BC2:]] = OpBitcast %[[#PTR_INT32]] %[[#GEP]]
; CHECK: %[[#VAL:]] = OpLoad %[[#INT32]] %[[#BC2]]
; CHECK: OpStore %[[#]] %[[#VAL]]
; CHECK: OpFunctionEnd
define spir_kernel void @test_vector_gep_with_load(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <1 x i64> zeroinitializer
  %elem = extractelement <1 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}
