; RUN: opt -S -passes=spirv-legalize-resource-binding -mtriple=spirv1.6-vulkan1.3-library < %s | FileCheck %s

@.str.used = private unnamed_addr constant [5 x i8] c"used\00", align 1

; Verify heap-binding intrinsic calls are assigned bindings after bindings
; already used in descriptor set 0 and use the heap index as the array index.

; CHECK-DAG: @ResourceDescriptorHeap.str = private unnamed_addr constant [23 x i8] c"ResourceDescriptorHeap\00", align 1
; CHECK-DAG: @ResourceDescriptorHeap.1.str = private unnamed_addr constant [25 x i8] c"ResourceDescriptorHeap.1\00", align 1
; CHECK-DAG: @SamplerDescriptorHeap.str = private unnamed_addr constant [22 x i8] c"SamplerDescriptorHeap\00", align 1

; CHECK-LABEL: define void @main(
define void @main() local_unnamed_addr #0 {
entry:

; Resource bound at desc 0 binding 0
; CHECK: call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 0, i32 1, i32 0, ptr @.str.used)
  %used = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str.used)

; Heap resource - bound to resource array at desc 0 binding 1
; CHECK: [[HEAPHANDLE:%.*]] = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 1, i32 0, i32 7, ptr @ResourceDescriptorHeap.1.str)
  %heap_resource_type1 = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefromheap.tspirv.VulkanBuffer_a0i32_12_1t(i32 7)

; Heap resource of different type - also bound to desc 0 binding 1
; CHECK: call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 1, i32 0, i32 13, ptr @ResourceDescriptorHeap.str)
; CHECK: call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 1, i32 0, i32 19, ptr @ResourceDescriptorHeap.str)
  %heap_resource_type2 = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromheap.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 13)
  %heap_resource_type2_again = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromheap.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 19)

; Heap sampler - bound to desc 0 binding 2
; CHECK: call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 2, i32 0, i32 8, ptr @SamplerDescriptorHeap.str)
  %heap_sampler = call target("spirv.Sampler") @llvm.spv.resource.handlefromheap.tspirv.Samplert(i32 8)

; Counters for heap resources - bound to desc 0 binding 3
; CHECK: call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefrombinding{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) [[HEAPHANDLE]], i32 0, i32 3)
  %heap_resource_counter = call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromheap.tspirv.VulkanBuffer_i32_12_1t.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %heap_resource_type1)

  ret void
}

; CHECK-NOT: @llvm.spv.resource.handlefromheap
; CHECK-NOT: @llvm.spv.resource.counterhandlefromheap

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }