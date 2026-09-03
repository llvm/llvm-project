; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test G_PTRMASK with 32-bit mask on 64-bit SPIR-V target.
; The 32-bit mask should be zero-extended to 64-bit before the AND operation.

; CHECK-LABEL: Begin function test_ptrmask_i32_mask
; CHECK: %[[#PTR_PARAM:]] = OpFunctionParameter
; CHECK: %[[#MASK_PARAM:]] = OpFunctionParameter
; CHECK: %[[#OUT_PARAM:]] = OpFunctionParameter
; CHECK: %[[#MASK_64:]] = OpUConvert %[[#]] %[[#MASK_PARAM]]
; CHECK-NEXT: %[[#PTR_AS_INT:]] = OpConvertPtrToU %[[#]] %[[#PTR_PARAM]]
; CHECK-NEXT: %[[#MASKED_INT:]] = OpBitwiseAnd %[[#]] %[[#PTR_AS_INT]] %[[#MASK_64]]
; CHECK-NEXT: %[[#MASKED_PTR:]] = OpConvertUToPtr %[[#]] %[[#MASKED_INT]]
; CHECK-NEXT: OpStore %[[#OUT_PARAM]] %[[#MASKED_PTR]] Aligned 8

define spir_kernel void @test_ptrmask_i32_mask(ptr addrspace(1) %ptr, i32 %mask, ptr addrspace(1) %out) {
entry:
  %mask_64 = zext i32 %mask to i64
  %masked_ptr = call ptr addrspace(1) @llvm.ptrmask.p1.i64(ptr addrspace(1) %ptr, i64 %mask_64)
  store ptr addrspace(1) %masked_ptr, ptr addrspace(1) %out, align 8
  ret void
}

declare ptr addrspace(1) @llvm.ptrmask.p1.i64(ptr addrspace(1), i64)
