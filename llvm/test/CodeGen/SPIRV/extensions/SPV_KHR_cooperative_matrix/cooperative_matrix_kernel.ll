; A GLCompute kernel that runs a cooperative-matrix matmul tile against
; descriptor-bound storage buffers under the Vulkan flavor. A and B are read-only
; StorageBuffers and C a writable StorageBuffer, each a VulkanBuffer handle
; decorated DescriptorSet and Binding and accessed through
; llvm.spv.resource.handlefrombinding and getpointer. The kernel loads A (use 0)
; and B (use 1) as cooperative matrices, mul-adds into a zero accumulator C
; (use 2), and stores C. The module must pass spirv-val --target-env vulkan1.3.

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_vulkan_memory_model %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_vulkan_memory_model %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpCapability CooperativeMatrixKHR
; CHECK-DAG: OpExtension "SPV_KHR_cooperative_matrix"
; CHECK-DAG: OpEntryPoint GLCompute %[[#entry:]] "main"
; CHECK-DAG: OpDecorate %[[#A:]] DescriptorSet 0
; CHECK-DAG: OpDecorate %[[#A]] Binding 0
; CHECK-DAG: OpDecorate %[[#B:]] DescriptorSet 0
; CHECK-DAG: OpDecorate %[[#B]] Binding 1
; CHECK-DAG: OpDecorate %[[#C:]] DescriptorSet 0
; CHECK-DAG: OpDecorate %[[#C]] Binding 2
; CHECK: %[[#MA:]] = OpCooperativeMatrixLoadKHR
; CHECK: %[[#MB:]] = OpCooperativeMatrixLoadKHR
; CHECK: %[[#MC0:]] = OpCompositeConstruct
; CHECK: %[[#MC:]] = OpCooperativeMatrixMulAddKHR %[[#]] %[[#MA]] %[[#MB]] %[[#MC0]]
; CHECK: OpCooperativeMatrixStoreKHR %[[#]] %[[#MC]]

@.str.a = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str.b = private unnamed_addr constant [2 x i8] c"b\00", align 1
@.str.c = private unnamed_addr constant [2 x i8] c"c\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  ; A: read-only StorageBuffer, binding 0 -> MatrixA (use 0)
  %ha = tail call target("spirv.VulkanBuffer", [0 x float], 12, 0)
      @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_0t(
          i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str.a)
  %pa = tail call ptr addrspace(11)
      @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_0t(
          target("spirv.VulkanBuffer", [0 x float], 12, 0) %ha, i32 0)
  %a = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0)
      @llvm.spv.cooperative.matrix.load(ptr addrspace(11) %pa, i32 0, i32 16)

  ; B: read-only StorageBuffer, binding 1 -> MatrixB (use 1)
  %hb = tail call target("spirv.VulkanBuffer", [0 x float], 12, 0)
      @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_0t(
          i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.b)
  %pb = tail call ptr addrspace(11)
      @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_0t(
          target("spirv.VulkanBuffer", [0 x float], 12, 0) %hb, i32 0)
  %b = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 1)
      @llvm.spv.cooperative.matrix.load(ptr addrspace(11) %pb, i32 0, i32 16)

  ; C = A*B + 0  (accumulator, use 2), stored to writable StorageBuffer binding 2
  %c0 = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.spv.cooperative.matrix.splat(float 0.000000e+00)
  %c = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
      @llvm.spv.cooperative.matrix.muladd(
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0) %a,
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 1) %b,
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %c0)
  %hc = tail call target("spirv.VulkanBuffer", [0 x float], 12, 1)
      @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_1t(
          i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.c)
  %pc = tail call ptr addrspace(11)
      @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(
          target("spirv.VulkanBuffer", [0 x float], 12, 1) %hc, i32 0)
  call void @llvm.spv.cooperative.matrix.store(
          ptr addrspace(11) %pc,
          target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %c,
          i32 0, i32 16)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
