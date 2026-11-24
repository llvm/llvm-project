; Test that loading and storing a vector of pointers needs no pointer bitcast.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#GenPtr:]] = OpTypePointer Generic %[[#Int8]]
; CHECK-DAG: %[[#Vec4GenPtr:]] = OpTypeVector %[[#GenPtr]] 4
; CHECK-DAG: %[[#Vec4GenPtrPtr:]] = OpTypePointer CrossWorkgroup %[[#Vec4GenPtr]]

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
