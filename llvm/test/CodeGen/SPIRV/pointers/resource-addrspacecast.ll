; RUN: llc -verify-machineinstrs -O3 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O3 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

@.str = private unnamed_addr constant [3 x i8] c"B0\00", align 1

%struct.S = type { i32 }

; CHECK-DAG:                     %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:                   %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:                  %[[#uint_10:]] = OpConstant %[[#uint]] 10
; CHECK-DAG:   %[[#ptr_StorageBuffer_uint:]] = OpTypePointer StorageBuffer %[[#uint]]
; CHECK-DAG:                   %[[#struct:]] = OpTypeStruct %[[#uint]]
; CHECK-DAG: %[[#ptr_StorageBuffer_struct:]] = OpTypePointer StorageBuffer %[[#struct]]
; CHECK-DAG:                     %[[#rarr:]] = OpTypeRuntimeArray %[[#struct]]
; CHECK-DAG:              %[[#rarr_struct:]] = OpTypeStruct %[[#rarr]]
; CHECK-DAG:       %[[#spirv_VulkanBuffer:]] = OpTypePointer StorageBuffer %[[#rarr_struct]]

define void @main() "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" {
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
; CHECK:      %[[#resource:]] = OpVariable %[[#spirv_VulkanBuffer]] StorageBuffer

  %ptr = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %handle, i32 0)
; CHECK: %[[#a:]] = OpCopyObject %[[#spirv_VulkanBuffer]] %[[#resource]]
; CHECK: %[[#b:]] = OpAccessChain %[[#ptr_StorageBuffer_struct]] %[[#a:]] %[[#uint_0]] %[[#uint_0]]
; CHECK: %[[#c:]] = OpInBoundsAccessChain %[[#ptr_StorageBuffer_uint]] %[[#b:]] %[[#uint_0]]
  %casted = addrspacecast ptr addrspace(11) %ptr to ptr

; CHECK: OpStore %[[#c]] %[[#uint_10]] Aligned 4
  store i32 10, ptr %casted, align 4
  ret void
}

declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1), i32)
