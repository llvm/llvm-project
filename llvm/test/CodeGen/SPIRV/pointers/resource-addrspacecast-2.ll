; RUN: llc -verify-machineinstrs -O3 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O3 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

@.str = private unnamed_addr constant [3 x i8] c"B0\00", align 1

%S2 = type { { [10 x { i32, i32 } ] }, i32 }

; CHECK-DAG:                     %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:                   %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:                   %[[#uint_1:]] = OpConstant %[[#uint]] 1
; CHECK-DAG:                   %[[#uint_3:]] = OpConstant %[[#uint]] 3
; CHECK-DAG:                  %[[#uint_10:]] = OpConstant %[[#uint]] 10
; CHECK-DAG:                  %[[#uint_11:]] = OpConstant %[[#uint]] 11
; CHECK-DAG:   %[[#ptr_StorageBuffer_uint:]] = OpTypePointer StorageBuffer %[[#uint]]

; CHECK-DAG:       %[[#t_s2_s_a_s:]] = OpTypeStruct %[[#uint]] %[[#uint]]
; CHECK-DAG:       %[[#t_s2_s_a:]] = OpTypeArray %[[#t_s2_s_a_s]] %[[#uint_10]]
; CHECK-DAG:       %[[#t_s2_s:]] = OpTypeStruct %[[#t_s2_s_a]]
; CHECK-DAG:       %[[#t_s2:]] = OpTypeStruct %[[#t_s2_s]] %[[#uint]]

; CHECK-DAG: %[[#ptr_StorageBuffer_struct:]] = OpTypePointer StorageBuffer %[[#t_s2]]
; CHECK-DAG:                     %[[#rarr:]] = OpTypeRuntimeArray %[[#t_s2]]
; CHECK-DAG:              %[[#rarr_struct:]] = OpTypeStruct %[[#rarr]]
; CHECK-DAG:       %[[#spirv_VulkanBuffer:]] = OpTypePointer StorageBuffer %[[#rarr_struct]]

define void @main() "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" {
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x %S2], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_Ss_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
; CHECK:      %[[#resource:]] = OpVariable %[[#spirv_VulkanBuffer]] StorageBuffer

  %ptr = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_Ss_12_1t(target("spirv.VulkanBuffer", [0 x %S2], 12, 1) %handle, i32 0)
; CHECK: %[[#a:]] = OpCopyObject %[[#spirv_VulkanBuffer]] %[[#resource]]
; CHECK: %[[#b:]] = OpAccessChain %[[#ptr_StorageBuffer_struct]] %[[#a:]] %[[#uint_0]] %[[#uint_0]]
  %casted = addrspacecast ptr addrspace(11) %ptr to ptr

; CHECK: %[[#ptr2:]] = OpInBoundsAccessChain %[[#ptr_StorageBuffer_uint]] %[[#b:]] %[[#uint_0]] %[[#uint_0]] %[[#uint_3]] %[[#uint_1]]
  %ptr2 = getelementptr inbounds %S2, ptr %casted, i64 0, i32 0, i32 0, i32 3, i32 1

; CHECK: OpStore %[[#ptr2]] %[[#uint_10]] Aligned 4
  store i32 10, ptr %ptr2, align 4

; Another store, but this time using LLVM's ability to load the first element
; without an explicit GEP. The backend has to determine the ptr type and
; generate the appropriate access chain.
; CHECK: %[[#ptr3:]] = OpInBoundsAccessChain %[[#ptr_StorageBuffer_uint]] %[[#b:]] %[[#uint_0]] %[[#uint_0]] %[[#uint_0]] %[[#uint_0]]
; CHECK: OpStore %[[#ptr3]] %[[#uint_11]] Aligned 4
  store i32 11, ptr %casted, align 4
  ret void
}

declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_S2s_12_1t(target("spirv.VulkanBuffer", [0 x %S2], 12, 1), i32)
