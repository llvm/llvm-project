; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"

@.str.1 = private unnamed_addr constant [2 x i8] c"B\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"S\00", align 1

; CHECK-DAG: [[int:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[zero:%[0-9]+]] = OpConstant [[int]] 0

; CHECK-DAG: [[ArrayType:%.+]] = OpTypeRuntimeArray [[int]]
; CHECK-DAG: [[BufferType:%.+]] = OpTypeStruct [[ArrayType]]
; CHECK-DAG: [[BufferPtrType:%.+]] = OpTypePointer StorageBuffer [[BufferType]]
; CHECK-DAG: [[BufferVar:%.+]] = OpVariable [[BufferPtrType]] StorageBuffer

; CHECK-DAG: [[StructType:%.+]] = OpTypeStruct [[int]] [[int]]
; CHECK-DAG: [[StructWrapper:%.+]] = OpTypeStruct [[StructType]]
; CHECK-DAG: [[StructPtrType:%.+]] = OpTypePointer StorageBuffer [[StructWrapper]]
; CHECK-DAG: [[StructVar:%.+]] = OpVariable [[StructPtrType]] StorageBuffer

define i32 @main() local_unnamed_addr {
entry:
; CHECK-DAG: [[BufferHandle:%.+]] = OpCopyObject [[BufferPtrType]] [[BufferVar]]
; CHECK-DAG: [[StructHandle:%.+]] = OpCopyObject [[StructPtrType]] [[StructVar]]

  %BufferHandle = tail call target("spirv.VulkanBuffer", [0 x i32], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.1)
  %StructHandle = tail call target("spirv.VulkanBuffer", {i32, i32}, 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBufferStruct(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.2)

; CHECK: [[AC2:%.+]] = OpAccessChain {{.*}} [[BufferHandle]] [[zero]]
; CHECK-NOT: [[AC2]] = OpAccessChain {{.*}} [[BufferHandle]] [[zero]] %
  %2 = tail call noundef nonnull align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getbasepointer.p11.tspirv.VulkanBuffer(target("spirv.VulkanBuffer", [0 x i32], 12, 0) %BufferHandle)
  %load2 = load i32, ptr addrspace(11) %2

; CHECK: [[AC3:%.+]] = OpAccessChain {{.*}} [[BufferHandle]] [[zero]] [[index:%[0-9]+]]
  %3 = tail call noundef nonnull align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer(target("spirv.VulkanBuffer", [0 x i32], 12, 0) %BufferHandle, i32 42)
  %load3 = load i32, ptr addrspace(11) %3

; CHECK: [[AC4:%.+]] = OpAccessChain {{.*}} [[StructHandle]] [[zero]]
; CHECK-NOT: [[AC4]] = OpAccessChain {{.*}} [[StructHandle]] [[zero]] %
  %4 = tail call noundef nonnull align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getbasepointer.p11.tspirv.VulkanBufferStruct(target("spirv.VulkanBuffer", {i32, i32}, 12, 0) %StructHandle)
  %load4 = load i32, ptr addrspace(11) %4

  %res1 = add i32 %load2, %load3
  %res2 = add i32 %res1, %load4
  ret i32 %res2
}

declare target("spirv.VulkanBuffer", [0 x i32], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer(i32, i32, i32, i32, ptr)
declare target("spirv.VulkanBuffer", {i32, i32}, 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBufferStruct(i32, i32, i32, i32, ptr)

declare ptr addrspace(11) @llvm.spv.resource.getbasepointer.p11.tspirv.VulkanBuffer(target("spirv.VulkanBuffer", [0 x i32], 12, 0))
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer(target("spirv.VulkanBuffer", [0 x i32], 12, 0), i32)
declare ptr addrspace(11) @llvm.spv.resource.getbasepointer.p11.tspirv.VulkanBufferStruct(target("spirv.VulkanBuffer", {i32, i32}, 12, 0))
