; Test that ptrtoint and inttoptr on vectors of pointers are correctly lowered
; to OpConvertPtrToU and OpConvertUToPtr when SPV_INTEL_masked_gather_scatter
; extension is enabled, and that loading and storing a vector of pointers
; needs no pointer bitcast.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability MaskedGatherScatterINTEL
; CHECK: OpExtension "SPV_INTEL_masked_gather_scatter"

; CHECK-DAG: %[[#Int64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PtrTy:]] = OpTypePointer CrossWorkgroup
; CHECK-DAG: %[[#Vec2Ptr:]] = OpTypeVector %[[#PtrTy]] 2
; CHECK-DAG: %[[#Vec2Int64:]] = OpTypeVector %[[#Int64]] 2
; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#GenPtr:]] = OpTypePointer Generic %[[#Int8]]
; CHECK-DAG: %[[#Vec4GenPtr:]] = OpTypeVector %[[#GenPtr]] 4
; CHECK-DAG: %[[#Vec4GenPtrPtr:]] = OpTypePointer CrossWorkgroup %[[#Vec4GenPtr]]

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

define spir_kernel void @test_load_store_vec4(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
; CHECK-LABEL: Begin function test_load_store_vec4
; CHECK: %[[#Src:]] = OpFunctionParameter %[[#Vec4GenPtrPtr]]
; CHECK: %[[#Dst:]] = OpFunctionParameter %[[#Vec4GenPtrPtr]]
; CHECK: %[[#V:]] = OpLoad %[[#Vec4GenPtr]] %[[#Src]]
; CHECK-NOT: OpBitcast
; CHECK: OpStore %[[#Dst]] %[[#V]]
; CHECK: OpFunctionEnd
entry:
  %v = load <4 x ptr addrspace(4)>, ptr addrspace(1) %src, align 8
  store <4 x ptr addrspace(4)> %v, ptr addrspace(1) %dst, align 8
  ret void
}
