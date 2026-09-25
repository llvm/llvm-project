; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val --target-env spv1.6 %}

; This test declares storage buffer Buf at descriptor set 0, binding 1, and
; creates one CBV, SRV, and UAV resource from the descriptor heap. Legalization
; creates a distinct runtime-array type for each heap resource type. All three
; arrays share descriptor set 0, binding 0 (the first available bindings).

; ModuleID = 'dyn-res-cvb-srv-uav.hlsl'
target datalayout = "e-ve-i64:64-n8:16:32:64-G10"

%S = type <{ <4 x i32> }>
@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

; CHECK-DAG: OpCapability RuntimeDescriptorArrayEXT

; CHECK-DAG: OpName [[Buf:%[0-9]+]] "Buf"
; CHECK-DAG: OpDecorate [[Buf]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[Buf]] Binding 1

; CHECK-DAG: OpName [[Heap0:%[0-9]+]] "ResourceDescriptorHeap"
; CHECK-DAG: OpDecorate [[Heap0]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[Heap0]] Binding 0

; CHECK-DAG: OpName [[Heap1:%[0-9]+]] "ResourceDescriptorHeap.1"
; CHECK-DAG: OpDecorate [[Heap1]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[Heap1]] Binding 0

; CHECK-DAG: OpName [[Heap2:%[0-9]+]] "ResourceDescriptorHeap.2"
; CHECK-DAG: OpDecorate [[Heap2]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[Heap2]] Binding 0

; Types of ResourceDescriptorHeap arrays

; CHECK: [[Int32:%[0-9]+]] = OpTypeInt 32 0
; CHECK: [[Int32Vec:%[0-9]+]] = OpTypeVector [[Int32]] 4
; CHECK: [[StructS:%[0-9]+]] = OpTypeStruct [[Int32Vec]]
; CHECK: [[ImageType:%[0-9]+]] = OpTypeImage [[Int32]] Buffer 2 0 0 2 R32i
; CHECK: [[RTArrayInt32Vec:%[0-9]+]] = OpTypeRuntimeArray [[Int32Vec]]
; CHECK: [[Heap1Struct:%[0-9]+]] = OpTypeStruct [[RTArrayInt32Vec]]
; CHECK: [[Heap0Struct:%[0-9]+]] = OpTypeStruct [[StructS]]
; CHECK: [[RTArrayImage:%[0-9]+]] = OpTypeRuntimeArray [[ImageType]]
; CHECK: [[Heap0RTArrayPtr:%[0-9]+]] = OpTypePointer UniformConstant [[RTArrayImage]]
; CHECK: [[Heap1RTArray:%[0-9]+]] = OpTypeRuntimeArray [[Heap1Struct]]
; CHECK: [[Heap1RTArrayPtr:%[0-9]+]] = OpTypePointer StorageBuffer [[Heap1RTArray]]
; CHECK: [[Heap0RTArray:%[0-9]+]] = OpTypeRuntimeArray [[Heap0Struct]]
; CHECK: [[Heap2RTArrayPtr:%[0-9]+]] = OpTypePointer Uniform [[Heap0RTArray]]

; CHECK: [[Heap2:%[0-9]+]] = OpVariable [[Heap2RTArrayPtr]] Uniform
; CHECK: [[Heap1:%[0-9]+]] = OpVariable [[Heap1RTArrayPtr]] StorageBuffer
; CHECK: [[Heap0:%[0-9]+]] = OpVariable [[Heap0RTArrayPtr]] UniformConstant

define void @test(i32 %CbvIndex, i32 %SrvIndex, i32 %UavIndex) {
entry:
  %Buf = tail call target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4i32_12_1t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str)

  %TId = tail call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  
  %HeapCvb = tail call target("spirv.VulkanBuffer", %S, 2, 0) @llvm.spv.resource.handlefromheap.tspirv.VulkanBuffer_s_Ss_2_0t(i32 %CbvIndex)
  %HeapSrv = tail call target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 0) @llvm.spv.resource.handlefromheap.tspirv.VulkanBuffer_a0v4i32_12_0t(i32 %SrvIndex)
  %HeapUav = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefromheap.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 %UavIndex)

  %CvbPtr = call noundef align 1 dereferenceable(16) ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Ss_2_0t(target("spirv.VulkanBuffer", %S, 2, 0) %HeapCvb)
  %CvbVal = load <4 x i32>, ptr addrspace(12) %CvbPtr, align 4
  
  %SrvPtr = call noundef align 4 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4i32_12_0t.i32(target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 0) %HeapSrv, i32 %TId)
  %SrvVal = load <4 x i32>, ptr addrspace(11) %SrvPtr, align 4
 
  %UavPtr = call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %HeapUav, i32 %TId)
  %UavVal = load i32, ptr addrspace(11) %UavPtr, align 4
 
  %0 = add <4 x i32> %SrvVal, %CvbVal
  %1 = insertelement <4 x i32> %0, i32 %UavVal, i64 0

  %BufPtr = call noundef align 4 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1) %Buf, i32 %TId)
  store <4 x i32> %1, ptr addrspace(11) %BufPtr, align 4

  ret void
}
