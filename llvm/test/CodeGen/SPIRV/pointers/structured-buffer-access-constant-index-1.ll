; RUN: llc -verify-machineinstrs -O3 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O3 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

%struct.S1 = type { <4 x i32>, [10 x <4 x float>], <4 x float> }
%struct.S2 = type { <4 x float>, <4 x i32> }

@.str = private unnamed_addr constant [3 x i8] c"In\00", align 1

define <4 x float> @main() {
entry:
  %0 = tail call target("spirv.VulkanBuffer", [0 x %struct.S1], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.S1s_12_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str)
  %3 = tail call noundef align 1 dereferenceable(192) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.S1s_12_0t(target("spirv.VulkanBuffer", [0 x %struct.S1], 12, 0) %0, i32 0)

; CHECK-DAG:  %[[#ulong:]] = OpTypeInt 64 0
; CHECK-DAG:  %[[#ulong_1:]] = OpConstant %[[#ulong]] 1
; CHECK-DAG:  %[[#ulong_3:]] = OpConstant %[[#ulong]] 3

; CHECK-DAG:  %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:  %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:  %[[#uint_10:]] = OpConstant %[[#uint]] 10

; CHECK-DAG:  %[[#float:]] = OpTypeFloat 32
; CHECK-DAG:  %[[#v4f:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG:  %[[#arr_v4f:]] = OpTypeArray %[[#v4f]] %[[#uint_10]]
; CHECK-DAG:  %[[#S1:]] = OpTypeStruct %[[#]] %[[#arr_v4f]] %[[#]]
; CHECK-DAG:  %[[#sb_S1:]] = OpTypePointer StorageBuffer %[[#S1]]
; CHECK-DAG:  %[[#sb_v4f:]] = OpTypePointer StorageBuffer %[[#v4f]]

; CHECK:      %[[#tmp:]] = OpAccessChain %[[#sb_S1]] %[[#]] %[[#uint_0]] %[[#uint_0]]
; CHECK:      %[[#ptr:]] = OpInBoundsAccessChain %[[#sb_v4f]] %[[#tmp]] %[[#ulong_1]] %[[#ulong_3]]
; This rewritten GEP combined all constant indices into a single value.
; We should make sure the correct indices are retrieved.
  %arrayidx.i = getelementptr inbounds nuw i8, ptr addrspace(11) %3, i64 64

; CHECK:  OpLoad %[[#v4f]] %[[#ptr]]
  %4 = load <4 x float>, ptr addrspace(11) %arrayidx.i, align 1

  ret <4 x float> %4
}

declare i32 @llvm.spv.flattened.thread.id.in.group()
declare target("spirv.VulkanBuffer", [0 x %struct.S1], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.S1s_12_0t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.S1s_12_0t(target("spirv.VulkanBuffer", [0 x %struct.S1], 12, 0), i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

