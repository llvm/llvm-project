; RUN: opt -S -passes=spirv-lower-resource-names,spirv-legalize-resource-binding -mtriple=spirv1.6-vulkan1.3-library < %s | FileCheck %s

@.str.used = private unnamed_addr constant [5 x i8] c"used\00", align 1

; Verify heap-binding intrinsic calls are assigned bindings after bindings
; already used in descriptor set 0 and use the heap index as the array index.

; CHECK-LABEL: define void @main(
define void @main() local_unnamed_addr #0 {
entry:

; Resource bound at desc 0 binding 0
; CHECK: call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.mdname{{.*}}(i32 0, i32 0, i32 1, i32 0, metadata
  %used = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str.used)

; Heap resource - bound to resource array at desc 0 binding 1
; CHECK: [[HEAPHANDLE:%.*]] = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.mdname{{.*}}(i32 0, i32 1, i32 0, i32 7, metadata ![[BUFFER:[0-9]+]])
  %heap_resource_type1 = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefromheap.tspirv.VulkanBuffer_a0i32_12_1t(i32 7)

; Heap resource of different type - also bound to desc 0 binding 1
; CHECK: call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.mdname{{.*}}(i32 0, i32 1, i32 0, i32 13, metadata ![[IMAGE:[0-9]+]])
; CHECK: call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.mdname{{.*}}(i32 0, i32 1, i32 0, i32 19, metadata ![[IMAGE]])
  %heap_resource_type2 = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromheap.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 13)
  %heap_resource_type2_again = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromheap.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 19)

; Heap sampler - bound to desc 0 binding 2
; CHECK: call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.mdname{{.*}}(i32 0, i32 2, i32 0, i32 8, metadata ![[SAMPLER:[0-9]+]])
  %heap_sampler = call target("spirv.Sampler") @llvm.spv.resource.handlefromheap.tspirv.Samplert(i32 8)

; Counters for heap resources - bound to desc 0 binding 3
; CHECK: call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefrombinding{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) [[HEAPHANDLE]], i32 0, i32 3)
  %heap_resource_counter = call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromheap.tspirv.VulkanBuffer_i32_12_1t.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %heap_resource_type1)

  ret void
}

; CHECK-NOT: @llvm.spv.resource.handlefromheap
; CHECK-NOT: @llvm.spv.resource.counterhandlefromheap

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

; CHECK-DAG: ![[BUFFER]] = !{!".ResourceDescriptorHeap.1"}
; CHECK-DAG: ![[IMAGE]] = !{!".ResourceDescriptorHeap.0"}
; CHECK-DAG: ![[SAMPLER]] = !{!".SamplerDescriptorHeap.0"}
