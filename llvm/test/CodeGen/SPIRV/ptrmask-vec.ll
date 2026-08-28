; RUN: llc -O0 --spirv-ext=+SPV_INTEL_masked_gather_scatter -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_INTEL_masked_gather_scatter -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that G_PTRMASK works with vector of pointers.
; This requires the SPV_INTEL_masked_gather_scatter extension.

; CHECK-DAG: OpCapability MaskedGatherScatterINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_masked_gather_scatter"
; CHECK-DAG: %[[#INT8_TY:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PTR_TY:]] = OpTypePointer CrossWorkgroup %[[#INT8_TY]]
; CHECK-DAG: %[[#VEC_PTR_TY:]] = OpTypeVector %[[#PTR_TY]] 2
; CHECK-DAG: %[[#INT64_TY:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#VEC_INT64_TY:]] = OpTypeVector %[[#INT64_TY]] 2
; CHECK: %[[#PTR_VEC_PARAM:]] = OpFunctionParameter %[[#VEC_PTR_TY]]
; CHECK: %[[#MASK_VEC_PARAM:]] = OpFunctionParameter %[[#VEC_INT64_TY]]
; CHECK: %[[#OUT_PARAM:]] = OpFunctionParameter
; CHECK: %[[#PTR_AS_INT:]] = OpConvertPtrToU %[[#VEC_INT64_TY]] %[[#PTR_VEC_PARAM]]
; CHECK: %[[#MASKED_INT:]] = OpBitwiseAnd %[[#VEC_INT64_TY]] %[[#PTR_AS_INT]] %[[#MASK_VEC_PARAM]]
; CHECK: %[[#MASKED_PTR:]] = OpConvertUToPtr %[[#VEC_PTR_TY]] %[[#MASKED_INT]]
; CHECK: %[[#ELEM0:]] = OpCompositeExtract %[[#PTR_TY]] %[[#MASKED_PTR]] 0
; CHECK: OpStore %[[#OUT_PARAM]] %[[#ELEM0]] Aligned 8

define spir_kernel void @test_ptrmask_vec(<2 x ptr addrspace(1)> %ptr_vec, <2 x i64> %mask_vec, ptr addrspace(1) %out) {
entry:
  %masked_ptr_vec = call <2 x ptr addrspace(1)> @llvm.ptrmask.v2p1.v2i64(<2 x ptr addrspace(1)> %ptr_vec, <2 x i64> %mask_vec)
  %elem0 = extractelement <2 x ptr addrspace(1)> %masked_ptr_vec, i32 0
  store ptr addrspace(1) %elem0, ptr addrspace(1) %out, align 8
  ret void
}

declare <2 x ptr addrspace(1)> @llvm.ptrmask.v2p1.v2i64(<2 x ptr addrspace(1)>, <2 x i64>)
