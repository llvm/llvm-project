; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; ModuleID = 'test_counters.hlsl'
source_filename = "test_counters.hlsl"

; CHECK: OpCapability Int8
; CHECK-DAG: OpName [[OutputBuffer:%[0-9]+]] "OutputBuffer"
; CHECK-DAG: OpName [[InputBuffer:%[0-9]+]] "InputBuffer"
; CHECK-DAG: OpName [[OutputBufferCounter:%[0-9]+]] "OutputBuffer.counter"
; CHECK-DAG: OpName [[InputBufferCounter:%[0-9]+]] "InputBuffer.counter"
; CHECK-DAG: OpDecorate [[OutputBuffer]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[OutputBuffer]] Binding 10
; CHECK-DAG: OpDecorate [[OutputBufferCounter]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[OutputBufferCounter]] Binding 0
; CHECK-DAG: OpDecorate [[InputBuffer]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[InputBuffer]] Binding 1
; CHECK-DAG: OpDecorate [[InputBufferCounter]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[InputBufferCounter]] Binding 2
; CHECK-DAG: [[int:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[zero:%[0-9]+]] = OpConstant [[int]] 0{{$}}
; CHECK-DAG: [[one:%[0-9]+]] = OpConstant [[int]] 1{{$}}
; CHECK-DAG: [[minus_one:%[0-9]+]] = OpConstant [[int]] 4294967295
; CHECK: [[OutputBufferHandle:%[0-9]+]] = OpCopyObject {{%[0-9]+}} [[OutputBuffer]]
; CHECK: [[InputBufferHandle:%[0-9]+]] = OpCopyObject {{%[0-9]+}} [[InputBuffer]]
; CHECK: [[InputCounterAC:%[0-9]+]] = OpAccessChain {{%[0-9]+}} [[InputBufferCounter]] [[zero]]
; CHECK: [[dec:%[0-9]+]] = OpAtomicIAdd [[int]] [[InputCounterAC]] [[one]] [[zero]] [[minus_one]]
; CHECK: [[iadd:%[0-9]+]] = OpIAdd [[int]] [[dec]] [[minus_one]]
; CHECK: [[OutputCounterAC:%[0-9]+]] = OpAccessChain {{%[0-9]+}} [[OutputBufferCounter]] [[zero]]
; CHECK: [[inc:%[0-9]+]] = OpAtomicIAdd [[int]] [[OutputCounterAC]] [[one]] [[zero]] [[one]]
; CHECK: [[InputAC:%[0-9]+]] = OpAccessChain {{%[0-9]+}} [[InputBufferHandle]] [[zero]] [[iadd]]
; CHECK: [[load:%[0-9]+]] = OpLoad {{%[0-9]+}} [[InputAC]]
; CHECK: [[OutputAC:%[0-9]+]] = OpAccessChain {{%[0-9]+}} [[OutputBufferHandle]] [[zero]] [[inc]]
; CHECK: OpStore [[OutputAC]] [[load]]


target triple = "spirv1.6-unknown-vulkan1.3-compute"

@.str = private unnamed_addr constant [13 x i8] c"OutputBuffer\00"
@.str.2 = private unnamed_addr constant [12 x i8] c"InputBuffer\00"

define void @main() #0 {
entry:
  %0 = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_1t(i32 0, i32 10, i32 1, i32 0, ptr @.str)
  %1 = call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromimplicitbinding.tspirv.VulkanBuffer_i32_12_1t.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %0, i32 0, i32 0)
  %2 = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_a0f32_12_1t(i32 1, i32 0, i32 1, i32 0, ptr @.str.2)
  %3 = call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromimplicitbinding.tspirv.VulkanBuffer_i32_12_1t.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %2, i32 2, i32 0)
  %4 = call i32 @llvm.spv.resource.updatecounter.tspirv.VulkanBuffer_i32_12_1t(target("spirv.VulkanBuffer", i32, 12, 1) %3, i8 -1)
  %5 = call i32 @llvm.spv.resource.updatecounter.tspirv.VulkanBuffer_i32_12_1t(target("spirv.VulkanBuffer", i32, 12, 1) %1, i8 1)
  %6 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %2, i32 %4)
  %7 = load float, ptr addrspace(11) %6
  %8 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %0, i32 %5)
  store float %7, ptr addrspace(11) %8
  ret void
}

declare target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_1t(i32, i32, i32, i32, ptr) #1
declare target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromimplicitbinding.tspirv.VulkanBuffer_i32_12_1t.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1), i32, i32) #1
declare target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_a0f32_12_1t(i32, i32, i32, i32, ptr) #1
declare i32 @llvm.spv.resource.updatecounter.tspirv.VulkanBuffer_i32_12_1t(target("spirv.VulkanBuffer", i32, 12, 1), i8) #2
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1), i32) #1

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="1,1,1" }
attributes #1 = { memory(none) }
attributes #2 = { memory(argmem: readwrite, inaccessiblemem: readwrite) }
