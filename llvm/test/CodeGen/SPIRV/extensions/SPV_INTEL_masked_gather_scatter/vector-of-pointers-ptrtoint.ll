; Test that ptrtoint and inttoptr on vectors of pointers are correctly lowered
; to OpConvertPtrToU and OpConvertUToPtr when SPV_INTEL_masked_gather_scatter
; extension is enabled.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; TODO: spirv-val does not support vector operands in OpConvertPtrToU and OpConvertUToPtr with SPV_INTEL_masked_gather_scatter
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability MaskedGatherScatterINTEL
; CHECK: OpExtension "SPV_INTEL_masked_gather_scatter"

; CHECK-DAG: %[[#Int64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PtrTy:]] = OpTypePointer CrossWorkgroup
; CHECK-DAG: %[[#Vec2Ptr:]] = OpTypeVector %[[#PtrTy]] 2
; CHECK-DAG: %[[#Vec2Int64:]] = OpTypeVector %[[#Int64]] 2

declare spir_func void @foo(<2 x i64>)

define spir_kernel void @test_ptrtoint_vec2(<2 x ptr addrspace(1)> %p) {
; CHECK-LABEL: Begin function test_ptrtoint_vec2
; CHECK: OpConvertPtrToU %[[#Vec2Int64]]
entry:
  %addr = ptrtoint <2 x ptr addrspace(1)> %p to <2 x i64>
  call void @foo(<2 x i64> %addr)
  ret void
}

declare spir_func void @bar(<2 x ptr addrspace(1)>)

define spir_kernel void @test_inttoptr_vec2(<2 x i64> %addr) {
; CHECK-LABEL: Begin function test_inttoptr_vec2
; CHECK: OpConvertUToPtr %[[#Vec2Ptr]]
entry:
  %p = inttoptr <2 x i64> %addr to <2 x ptr addrspace(1)>
  call void @bar(<2 x ptr addrspace(1)> %p)
  ret void
}
