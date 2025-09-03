; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

@.str = private unnamed_addr constant [4 x i8] c"In3\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"Out4\00", align 1
@.str.3 = private unnamed_addr constant [5 x i8] c"Out3\00", align 1


; CHECK-DAG: %[[#INT32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#INT4:]] = OpTypeVector %[[#INT32]] 4
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FLOAT4:]] = OpTypeVector %[[#FLOAT]] 4
; CHECK-DAG: %[[#INT3:]] = OpTypeVector %[[#INT32]] 3
; CHECK-DAG: %[[#UNDEF_INT4:]] = OpUndef %[[#INT4]]

define void @case1() local_unnamed_addr {
  ; CHECK: %[[#BUFFER_LOAD:]] = OpLoad %[[#FLOAT4]] %{{[0-9]+}} Aligned 16
  ; CHECK: %[[#CAST_LOAD:]] = OpBitcast %[[#INT4]] %[[#BUFFER_LOAD]]
  ; CHECK: %[[#VEC_SHUFFLE:]] = OpVectorShuffle %[[#INT4]] %[[#CAST_LOAD]] %[[#CAST_LOAD]] 0 1 2 3
  %1 = tail call target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_0t(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str)
  %2 = tail call target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4i32_12_1t(i32 0, i32 5, i32 1, i32 0, ptr nonnull @.str.2)
  %3 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_0t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 0) %1, i32 0)
  %4 = load <4 x i32>, ptr addrspace(11) %3, align 16
  %5 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4i32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1) %2, i32 0)
  store <4 x i32> %4, ptr addrspace(11) %5, align 16
  ret void
}

define void @case2() local_unnamed_addr {
  ; CHECK: %[[#BUFFER_LOAD:]] = OpLoad %[[#FLOAT4]] %{{[0-9]+}} Aligned 16
  ; CHECK: %[[#CAST_LOAD:]] = OpBitcast %[[#INT4]] %[[#BUFFER_LOAD]]
  ; CHECK: %[[#VEC_SHUFFLE:]] = OpVectorShuffle %[[#INT4]] %[[#CAST_LOAD]] %[[#CAST_LOAD]] 0 1 2 3
  ; CHECK: %[[#VEC_TRUNCATE:]] = OpVectorShuffle %[[#INT3]] %[[#VEC_SHUFFLE]] %[[#UNDEF_INT4]] 0 1 2
  %1 = tail call target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_0t(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str)
  %2 = tail call target("spirv.VulkanBuffer", [0 x <3 x i32>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v3i32_12_1t(i32 0, i32 5, i32 1, i32 0, ptr nonnull @.str.3)
  %3 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_0t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 0) %1, i32 0)
  %4 = load <4 x i32>, ptr addrspace(11) %3, align 16
  %5 = shufflevector <4 x i32> %4, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %6 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v3i32_12_1t(target("spirv.VulkanBuffer", [0 x <3 x i32>], 12, 1) %2, i32 0)
  store <3 x i32> %5, ptr addrspace(11) %6, align 16
  ret void
}
