; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:        %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:     %[[#v2_uint:]] = OpTypeVector %[[#uint]] 2
; CHECK-DAG:      %[[#double:]] = OpTypeFloat 64
; CHECK-DAG:   %[[#v2_double:]] = OpTypeVector %[[#double]] 2
; CHECK-DAG:     %[[#v4_uint:]] = OpTypeVector %[[#uint]] 4
@.str = private unnamed_addr constant [3 x i8] c"In\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"Out\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  %0 = tail call target("spirv.VulkanBuffer", [0 x <2 x i32>], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v2i32_12_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %1 = tail call target("spirv.VulkanBuffer", [0 x <2 x double>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v2f64_12_1t(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.2)
  %2 = tail call noundef align 8 dereferenceable(8) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v2i32_12_0t(target("spirv.VulkanBuffer", [0 x <2 x i32>], 12, 0) %0, i32 0)
  %3 = load <2 x i32>, ptr addrspace(11) %2, align 8
  %4 = tail call noundef align 8 dereferenceable(8) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v2i32_12_0t(target("spirv.VulkanBuffer", [0 x <2 x i32>], 12, 0) %0, i32 1)
  %5 = load <2 x i32>, ptr addrspace(11) %4, align 8
; CHECK: %[[#tmp:]] = OpVectorShuffle %[[#v4_uint]] {{%[0-9]+}} {{%[0-9]+}} 0 2 1 3
  %6 = shufflevector <2 x i32> %3, <2 x i32> %5, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
; CHECK: %[[#access:]] = OpAccessChain {{.*}}
  %7 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v2f64_12_1t(target("spirv.VulkanBuffer", [0 x <2 x double>], 12, 1) %1, i32 0)
; CHECK: %[[#bitcast:]] = OpBitcast %[[#v2_double]] %[[#tmp]]
; CHECK: OpStore %[[#access]] %[[#bitcast]] Aligned 16
  store <4 x i32> %6, ptr addrspace(11) %7, align 16
  ret void
}
