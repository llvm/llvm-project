; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#C0:]] = OpConstant %[[#UINT]] 0{{$}}
; CHECK-DAG: %[[#C1:]] = OpConstant %[[#UINT]] 1{{$}}
; CHECK-DAG: %[[#C2:]] = OpConstant %[[#UINT]] 2{{$}}
; CHECK-DAG: %[[#C3:]] = OpConstant %[[#UINT]] 3{{$}}
; CHECK-DAG: %[[#C4:]] = OpConstant %[[#UINT]] 4{{$}}

; CHECK-LABEL: Begin function main{{$}}
; CHECK: %[[#AC0:]] = OpAccessChain %[[#]] %[[#]] %[[#C0]] %[[#C0]]
; CHECK: OpStore %[[#AC0]]
; CHECK: %[[#AC1:]] = OpAccessChain %[[#]] %[[#]] %[[#C0]] %[[#C1]]
; CHECK: OpStore %[[#AC1]]
; CHECK: %[[#AC2:]] = OpAccessChain %[[#]] %[[#]] %[[#C0]] %[[#C2]]
; CHECK: OpStore %[[#AC2]]
; CHECK: %[[#AC3:]] = OpAccessChain %[[#]] %[[#]] %[[#C0]] %[[#C3]]
; CHECK: OpStore %[[#AC3]]

; CHECK-LABEL: Begin function main_dyn
; CHECK: %[[#TID:]] = OpCompositeExtract %[[#UINT]] %[[#]] 0
; CHECK: %[[#DAC0:]] = OpAccessChain %[[#]] %[[#]] %[[#C0]] %[[#TID]]
; CHECK: OpStore %[[#DAC0]]
; CHECK: %[[#OFF1:]] = OpIAdd %[[#UINT]] %[[#TID]] %[[#C1]]
; CHECK: %[[#DAC1:]] = OpAccessChain %[[#]] %[[#]] %[[#C0]] %[[#OFF1]]
; CHECK: OpStore %[[#DAC1]]
; CHECK: %[[#OFF2:]] = OpIAdd %[[#UINT]] %[[#TID]] %[[#C2]]
; CHECK: %[[#DAC2:]] = OpAccessChain %[[#]] %[[#]] %[[#C0]] %[[#OFF2]]
; CHECK: OpStore %[[#DAC2]]
; CHECK: %[[#OFF3:]] = OpIAdd %[[#UINT]] %[[#TID]] %[[#C3]]
; CHECK: %[[#DAC3:]] = OpAccessChain %[[#]] %[[#]] %[[#C0]] %[[#OFF3]]
; CHECK: OpStore %[[#DAC3]]
; CHECK: %[[#OFF4:]] = OpIAdd %[[#UINT]] %[[#TID]] %[[#C4]]
; CHECK: %[[#DAC4:]] = OpAccessChain %[[#]] %[[#]] %[[#C0]] %[[#OFF4]]
; CHECK: OpStore %[[#DAC4]]
; CHECK: OpIAdd %[[#UINT]] %[[#OFF4]] %[[#C1]]

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  store i32 42, ptr addrspace(11) %ptr, align 4
  ret void
}

define void @main_dyn() local_unnamed_addr #0 {
entry:
  %tid = tail call i32 @llvm.spv.thread.id.i32(i32 0)
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 %tid)
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr addrspace(11) %ptr, align 16
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
