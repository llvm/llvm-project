; RUN: llc -verify-machineinstrs -O0 --spirv-ext=+SPV_INTEL_masked_gather_scatter,+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; spirv-val has a bug around validating OpTypeVectorIdEXTs with Pointer type elements, as it seems to only allow scalar numerical types for the latter.
; TODO: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_INTEL_masked_gather_scatter,+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that G_PTRMASK works with vector of pointers.
; This requires the SPV_INTEL_masked_gather_scatter extension.

; CHECK-DAG: OpCapability MaskedGatherScatterINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_masked_gather_scatter"
; CHECK-DAG: OpCapability LongVectorEXT
; CHECK-DAG: OpExtension "SPV_EXT_long_vector"
; CHECK-DAG: %[[#INT64_TY:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#INT32_TY:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#INT8_TY:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PTR_TY:]] = OpTypePointer CrossWorkgroup %[[#INT8_TY]]
; CHECK-DAG: %[[#SEVENTEEN:]] = OpConstant %[[#INT32_TY]] 17
; CHECK-DAG: %[[#VEC_PTR_TY:]] = OpTypeVectorIdEXT %[[#PTR_TY]] %[[#SEVENTEEN]]
; CHECK-DAG: %[[#VEC_INT64_TY:]] = OpTypeVectorIdEXT %[[#INT64_TY]] %[[#SEVENTEEN]]
; CHECK: %[[#PTR_VEC_PARAM:]] = OpFunctionParameter %[[#VEC_PTR_TY]]
; CHECK: %[[#MASK_VEC_PARAM:]] = OpFunctionParameter %[[#VEC_INT64_TY]]
; CHECK: %[[#OUT_PARAM:]] = OpFunctionParameter
; CHECK: %[[#PTR_AS_INT:]] = OpConvertPtrToU %[[#VEC_INT64_TY]] %[[#PTR_VEC_PARAM]]
; CHECK: %[[#MASKED_INT:]] = OpBitwiseAnd %[[#VEC_INT64_TY]] %[[#PTR_AS_INT]] %[[#MASK_VEC_PARAM]]
; CHECK: %[[#MASKED_PTR:]] = OpConvertUToPtr %[[#VEC_PTR_TY]] %[[#MASKED_INT]]
; CHECK: %[[#ELEM0:]] = OpCompositeExtract %[[#PTR_TY]] %[[#MASKED_PTR]] 0
; CHECK: OpStore %[[#OUT_PARAM]] %[[#ELEM0]] Aligned 8

define spir_kernel void @test_ptrmask_vec(<17 x ptr addrspace(1)> %ptr_vec, <17 x i64> %mask_vec, ptr addrspace(1) %out) {
entry:
  %masked_ptr_vec = call <17 x ptr addrspace(1)> @llvm.ptrmask.v17p1.v17i64(<17 x ptr addrspace(1)> %ptr_vec, <17 x i64> %mask_vec)
  %elem0 = extractelement <17 x ptr addrspace(1)> %masked_ptr_vec, i32 0
  store ptr addrspace(1) %elem0, ptr addrspace(1) %out, align 8
  ret void
}

declare <17 x ptr addrspace(1)> @llvm.ptrmask.v17p1.v17i64(<17 x ptr addrspace(1)>, <17 x i64>)
