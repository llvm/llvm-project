; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[FLOAT:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[UINT:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[V4FLOAT:[0-9]+]] = OpTypeVector %[[FLOAT]] 4
; CHECK: OpBitcast %[[FLOAT]]
; CHECK: OpCompositeConstruct %[[V4FLOAT]]
; CHECK: OpExtInst %[[V4FLOAT]] {{.*}} {{[Cc]osh}}
; CHECK: OpBitcast %[[UINT]]

@.str = private unnamed_addr constant [4 x i8] c"In0\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"Out\00", align 1

declare token @llvm.experimental.convergence.entry()

define void @main() local_unnamed_addr #0 {
entry:
  %convergence = tail call token @llvm.experimental.convergence.entry()
  %input = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i8_12_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %output = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i8_12_1t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
  %input.ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_0t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %input, i32 0) [ "convergencectrl"(token %convergence) ]
  %value = load <16 x float>, ptr addrspace(11) %input.ptr, align 4
  %result = tail call <16 x float> @llvm.cosh.v16f32(<16 x float> %value)
  %output.ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %output, i32 0) [ "convergencectrl"(token %convergence) ]
  store <16 x float> %result, ptr addrspace(11) %output.ptr, align 4
  ret void
}

declare <16 x float> @llvm.cosh.v16f32(<16 x float>)

declare target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i8_12_0t(i32, i32, i32, i32, ptr)
declare target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i8_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_0t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 0), i32)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1), i32)

attributes #0 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
