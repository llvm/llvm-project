; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that G_PTRMASK is correctly lowered to SPIR-V for 64-bit targets.
; Result = Ptr & Mask
; This should generate: OpConvertPtrToU -> OpBitwiseAnd -> OpConvertUToPtr

; CHECK-LABEL: Begin function test_ptrmask_i64
; CHECK: %[[#PTR_PARAM:]] = OpFunctionParameter
; CHECK: %[[#MASK_PARAM:]] = OpFunctionParameter
; CHECK: %[[#OUT_PARAM:]] = OpFunctionParameter
; CHECK: %[[#PTR_AS_INT:]] = OpConvertPtrToU %[[#]] %[[#PTR_PARAM]]
; CHECK-NEXT: %[[#MASKED_INT:]] = OpBitwiseAnd %[[#]] %[[#PTR_AS_INT]] %[[#MASK_PARAM]]
; CHECK-NEXT: %[[#MASKED_PTR:]] = OpConvertUToPtr %[[#]] %[[#MASKED_INT]]
; CHECK-NEXT: OpStore %[[#OUT_PARAM]] %[[#MASKED_PTR]] Aligned 8

define spir_kernel void @test_ptrmask_i64(ptr addrspace(1) %ptr, i64 %mask, ptr addrspace(1) %out) {
entry:
  %masked_ptr = call ptr addrspace(1) @llvm.ptrmask.p1.i64(ptr addrspace(1) %ptr, i64 %mask)
  store ptr addrspace(1) %masked_ptr, ptr addrspace(1) %out, align 8
  ret void
}

declare ptr addrspace(1) @llvm.ptrmask.p1.i64(ptr addrspace(1), i64)
