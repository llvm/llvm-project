; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#C1:]] = OpConstant %[[#UINT]] 1{{$}}
; CHECK-DAG: %[[#C4:]] = OpConstant %[[#UINT]] 4{{$}}
; CHECK-DAG: [[AC:%[0-9]+]] = OpAccessChain {{.*}}
; CHECK: OpLoad {{.*}} [[AC]]
; CHECK-NOT: OpTypeVector
; CHECK: OpCompositeInsert
; CHECK: OpStore {{.*}}

; CHECK-LABEL: Begin function main_dyn
; CHECK: %[[#TID:]] = OpCompositeExtract %[[#UINT]] %[[#]] 0
; CHECK: %[[#DAC0:]] = OpAccessChain %[[#]] %[[#]] %[[#]] %[[#TID]]
; CHECK: OpLoad %[[#]] %[[#DAC0]]
; CHECK: %[[#OFF1:]] = OpIAdd %[[#UINT]] %[[#TID]] %[[#C1]]
; CHECK: %[[#DAC1:]] = OpAccessChain %[[#]] %[[#]] %[[#]] %[[#OFF1]]
; CHECK: OpLoad %[[#]] %[[#DAC1]]
; CHECK: %[[#OFF4:]] = OpIAdd %[[#UINT]] %[[#TID]] %[[#C4]]
; CHECK: %[[#DAC4:]] = OpAccessChain %[[#]] %[[#]] %[[#]] %[[#OFF4]]
; CHECK: OpLoad %[[#]] %[[#DAC4]]
; CHECK: OpIAdd %[[#UINT]] %[[#OFF4]] %[[#C1]]
; CHECK: OpCompositeInsert
; CHECK: OpStore

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1
@out = addrspace(10) global <4 x i32> zeroinitializer, align 16

define void @main() local_unnamed_addr #0 {
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  %val = load <4 x i32>, ptr addrspace(11) %ptr, align 16
  store <4 x i32> %val, ptr addrspace(10) @out, align 16
  ret void
}

define void @main_dyn() local_unnamed_addr #0 {
entry:
  %tid = tail call i32 @llvm.spv.thread.id.i32(i32 0)
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 %tid)
  %val = load <4 x i32>, ptr addrspace(11) %ptr, align 16
  store <4 x i32> %val, ptr addrspace(10) @out, align 16
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
