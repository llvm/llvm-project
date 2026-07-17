; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; spirv-val does not yet handle long vectors of pointers even though the rules are the same as for OpTypeVector
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability LongVectorEXT
; CHECK: OpExtension "SPV_EXT_long_vector"
; CHECK-DAG: %[[#INT32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR_INT32:]] = OpTypePointer CrossWorkgroup %[[#INT32]]
; CHECK-DAG: %[[#INT64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#VEC1_INT64:]] = OpTypeVectorIdEXT %[[#INT64]] 1
; CHECK-DAG: %[[#VEC1_PTR:]] = OpTypeVectorIdEXT %[[#PTR_INT32]] 1
; CHECK-DAG: %[[#NULL_VEC:]] = OpConstantNull %[[#VEC1_INT64]]
; CHECK-LABEL: Begin function test_vector_gep_with_load
; CHECK: %[[#GEP:]] = OpPtrAccessChain %[[#PTR_INT32]] %[[#]] %[[#NULL_VEC]]
; CHECK: %[[#INSERTELT:]] = OpCompositeInsert %[[#VEC1_PTR]] %[[#GEP]]
; CHECK: %[[#EXTRACTELT:]] = OpCompositeExtract %[[#PTR_INT32]] %[[#INSERTELT]]
; CHECK: %[[#VAL:]] = OpLoad %[[#INT32]] %[[#EXTRACTELT]]
; CHECK: OpStore %[[#]] %[[#VAL]]
; CHECK: OpFunctionEnd
define spir_kernel void @test_vector_gep_with_load(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <1 x i64> zeroinitializer
  %elem = extractelement <1 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}
