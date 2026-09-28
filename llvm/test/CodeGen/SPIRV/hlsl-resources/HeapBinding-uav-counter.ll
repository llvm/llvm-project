; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val --target-env spv1.6 %}

; This test declares storage buffer Buf at descriptor set 0, binding 1, and
; creates a UAV and its counter by dynamically indexing into descriptor heap.
; SPIR-V legalization creates separate runtime arrays for the resource and
; counter descriptors, assigning the resource heap to binding 0 and the
; counter heap to binding 2 (the first two available bindings).

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

; CHECK-DAG: OpCapability RuntimeDescriptorArrayEXT

; CHECK-DAG: OpName [[Buf:%[0-9]+]] "Buf"
; CHECK-DAG: OpDecorate [[Buf]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[Buf]] Binding 1

; CHECK-DAG: OpName [[Heap:%[0-9]+]] "ResourceDescriptorHeap"
; CHECK-DAG: OpDecorate [[Heap]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[Heap]] Binding 0

; CHECK-DAG: OpName [[CounterHeap:%[0-9]+]] "ResourceDescriptorHeap.counter"
; CHECK-DAG: OpDecorate [[CounterHeap]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[CounterHeap]] Binding 2

; Types of ResourceDescriptorHeap arrays

; CHECK: [[Int32:%[0-9]+]] = OpTypeInt 32 0
; CHECK: [[Float32:%[0-9]+]] = OpTypeFloat 32
; CHECK: [[RTArrayFloat:%[0-9]+]] = OpTypeRuntimeArray [[Float32]]
; CHECK: [[HeapStruct:%[0-9]+]] = OpTypeStruct [[RTArrayFloat]]
; CHECK: [[CounterStruct:%[0-9]+]] = OpTypeStruct [[Int32]]
; CHECK: [[CounterRTArray:%[0-9]+]] = OpTypeRuntimeArray [[CounterStruct]]
; CHECK: [[CounterRTArrayPtr:%[0-9]+]] = OpTypePointer StorageBuffer [[CounterRTArray]]
; CHECK: [[HeapRTArray:%[0-9]+]] = OpTypeRuntimeArray [[HeapStruct]]
; CHECK: [[HeapRTArrayPtr:%[0-9]+]] = OpTypePointer StorageBuffer [[HeapRTArray]]

; CHECK: [[Heap]] = OpVariable [[HeapRTArrayPtr]] StorageBuffer
; CHECK: [[CounterHeap]] = OpVariable [[CounterRTArrayPtr]] StorageBuffer

define void @test(i32 %Index) {
entry:
  %TId = tail call i32 @llvm.spv.thread.id.in.group.i32(i32 0)

  %HeapUav = tail call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefromheap.tspirv.VulkanBuffer_a0f32_12_1t(i32 %Index)
  %HeapUavCounter = call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromheap.tspirv.VulkanBuffer_i32_12_1t.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %HeapUav)
  
  %UavPtr = call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t.i32(target("spirv.VulkanBuffer", [0 x float], 12, 1) %HeapUav, i32 0)
  %UavVal = load float, ptr addrspace(11) %UavPtr, align 4

  %CounterVal = tail call noundef i32 @llvm.spv.resource.updatecounter.tspirv.VulkanBuffer_i32_12_1t(target("spirv.VulkanBuffer", i32, 12, 1) %HeapUavCounter, i8 1)
  %CounterValFloat = uitofp reassoc nnan ninf nsz arcp afn i32 %CounterVal to float

  %Buf = tail call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_1t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str)
  %BufPtr = call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t.i32(target("spirv.VulkanBuffer", [0 x float], 12, 1) %Buf, i32 0)

  %add = fadd reassoc nnan ninf nsz arcp afn float %CounterValFloat, %UavVal
  store float %add, ptr addrspace(11) %BufPtr, align 4
  ret void
}
