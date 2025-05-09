; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"

; CHECK-DAG: OpName [[ScalarBlock_var:%[0-9]+]] "__resource_p_12_{_u32[0]}_0_0"
; CHECK-DAG: OpName [[buffer_var:%[0-9]+]] "__resource_p_12_{_{_{_u32_f32[3]}[10]}[0]}_0_0"
; CHECK-DAG: OpName [[array_buffer_var:%[0-9]+]] "__resource_p_12_{_{_{_u32_f32[3]}[10]}[0]}[10]_0_0"

; CHECK-DAG: OpMemberDecorate [[ScalarBlock:%[0-9]+]] 0 Offset 0
; CHECK-DAG: OpDecorate [[ScalarBlock]] Block
; CHECK-DAG: OpMemberDecorate [[ScalarBlock]] 0 NonWritable
; CHECK-DAG: OpMemberDecorate [[T_explicit:%[0-9]+]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate [[T_explicit]] 1 Offset 16
; CHECK-DAG: OpDecorate [[T_array_explicit:%[0-9]+]] ArrayStride 32
; CHECK-DAG: OpMemberDecorate [[S_explicit:%[0-9]+]] 0 Offset 0
; CHECK-DAG: OpDecorate [[S_array_explicit:%[0-9]+]] ArrayStride 320
; CHECK-DAG: OpMemberDecorate [[block:%[0-9]+]] 0 Offset 0
; CHECK-DAG: OpDecorate [[block]] Block
; CHECK-DAG: OpMemberDecorate [[block]] 0 NonWritable

; CHECK-DAG: [[float:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[v3f:%[0-9]+]] = OpTypeVector [[float]] 3
; CHECK-DAG: [[uint:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[T:%[0-9]+]] = OpTypeStruct [[uint]] [[v3f]]
; CHECK-DAG: [[T_explicit]] = OpTypeStruct [[uint]] [[v3f]]
%struct.T = type { i32, <3 x float> }

; CHECK-DAG: [[zero:%[0-9]+]] = OpConstant [[uint]] 0{{$}}
; CHECK-DAG: [[one:%[0-9]+]] = OpConstant [[uint]] 1{{$}}
; CHECK-DAG: [[ten:%[0-9]+]] = OpConstant [[uint]] 10
; CHECK-DAG: [[T_array:%[0-9]+]] = OpTypeArray [[T]] [[ten]]
; CHECK-DAG: [[S:%[0-9]+]] = OpTypeStruct [[T_array]]
; CHECK-DAG: [[T_array_explicit]] = OpTypeArray [[T_explicit]] [[ten]]
; CHECK-DAG: [[S_explicit]] = OpTypeStruct [[T_array_explicit]]
%struct.S = type { [10 x %struct.T] }

; CHECK-DAG: [[private_S_ptr:%[0-9]+]] = OpTypePointer Private [[S]]
; CHECK-DAG: [[private_var:%[0-9]+]] = OpVariable [[private_S_ptr]] Private
@private = internal addrspace(10) global %struct.S poison

; CHECK-DAG: [[storagebuffer_S_ptr:%[0-9]+]] = OpTypePointer StorageBuffer [[S_explicit]]
; CHECK-DAG: [[storage_buffer:%[0-9]+]] = OpVariable [[storagebuffer_S_ptr]] StorageBuffer
@storage_buffer = internal addrspace(11) global %struct.S poison

; CHECK-DAG: [[storagebuffer_int_ptr:%[0-9]+]] = OpTypePointer StorageBuffer [[uint]]
; CHECK-DAG: [[ScalarBlock_array:%[0-9]+]] = OpTypeRuntimeArray [[uint]]
; CHECK-DAG: [[ScalarBlock]] = OpTypeStruct [[ScalarBlock_array]]
; CHECK-DAG: [[ScalarBlock_ptr:%[0-9]+]] = OpTypePointer StorageBuffer [[ScalarBlock]]
; CHECK-DAG: [[ScalarBlock_var]] = OpVariable [[ScalarBlock_ptr]] StorageBuffer


; CHECK-DAG: [[S_array_explicit]] = OpTypeRuntimeArray [[S_explicit]]
; CHECK-DAG: [[block]] = OpTypeStruct [[S_array_explicit]]
; CHECK-DAG: [[buffer_ptr:%[0-9]+]] = OpTypePointer StorageBuffer [[block]]
; CHECK-DAG: [[buffer_var]] = OpVariable [[buffer_ptr]] StorageBuffer

; CHECK-DAG: [[array_buffer:%[0-9]+]] = OpTypeArray [[block]] [[ten]]
; CHECK-DAG: [[array_buffer_ptr:%[0-9]+]] = OpTypePointer StorageBuffer [[array_buffer]]
; CHECK-DAG: [[array_buffer_var]] = OpVariable [[array_buffer_ptr]] StorageBuffer

; CHECK: OpFunction [[uint]] None
define external i32 @scalar_vulkan_buffer_load() {
; CHECK-NEXT: OpLabel
entry:
; CHECK-NEXT: [[handle:%[0-9]+]] = OpCopyObject [[ScalarBlock_ptr]] [[ScalarBlock_var]]
  %handle = tail call target("spirv.VulkanBuffer", [0 x i32], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false)

; CHECK-NEXT: [[ptr:%[0-9]+]] = OpAccessChain [[storagebuffer_int_ptr]] [[handle]] [[zero]] [[one]]
  %0 = tail call noundef nonnull align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 0) %handle, i32 1)

; CHECK-NEXT: [[ld:%[0-9]+]] = OpLoad [[uint]] [[ptr]] Aligned 4
  %1 = load i32, ptr addrspace(11) %0, align 4

; CHECK-NEXT: OpReturnValue [[ld]]
  ret i32 %1

; CHECK-NEXT: OpFunctionEnd
}

; CHECK: OpFunction [[S]] None
define external %struct.S @private_load() {
; CHECK-NEXT: OpLabel
entry:

; CHECK-NEXT: [[ld:%[0-9]+]] = OpLoad [[S]] [[private_var]] Aligned 4
  %1 = load %struct.S, ptr addrspace(10) @private, align 4

; CHECK-NEXT: OpReturnValue [[ld]]
  ret %struct.S %1

; CHECK-NEXT: OpFunctionEnd
}

; CHECK: OpFunction [[S]] None
define external %struct.S @storage_buffer_load() {
; CHECK-NEXT: OpLabel
entry:

; CHECK-NEXT: [[ld:%[0-9]+]] = OpLoad [[S_explicit]] [[storage_buffer]] Aligned 4
; CHECK-NEXT: [[copy:%[0-9]+]] = OpCopyLogical [[S]] [[ld]]
  %1 = load %struct.S, ptr addrspace(11) @storage_buffer, align 4

; CHECK-NEXT: OpReturnValue [[copy]]
  ret %struct.S %1

; CHECK-NEXT: OpFunctionEnd
}

; CHECK: OpFunction [[S]] None
define external %struct.S @vulkan_buffer_load() {
; CHECK-NEXT: OpLabel
entry:
; CHECK-NEXT: [[handle:%[0-9]+]] = OpCopyObject [[buffer_ptr]] [[buffer_var]]
  %handle = tail call target("spirv.VulkanBuffer", [0 x %struct.S], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false)

; CHECK-NEXT: [[ptr:%[0-9]+]] = OpAccessChain [[storagebuffer_S_ptr]] [[handle]] [[zero]] [[one]]
  %0 = tail call noundef nonnull align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 0) %handle, i32 1)

; CHECK-NEXT: [[ld:%[0-9]+]] = OpLoad [[S_explicit]] [[ptr]] Aligned 4
; CHECK-NEXT: [[copy:%[0-9]+]] = OpCopyLogical [[S]] [[ld]]
  %1 = load %struct.S, ptr addrspace(11) %0, align 4

; CHECK-NEXT: OpReturnValue [[copy]]
  ret %struct.S %1

; CHECK-NEXT: OpFunctionEnd
}

; CHECK: OpFunction [[S]] None
define external %struct.S @array_of_vulkan_buffers_load() {
; CHECK-NEXT: OpLabel
entry:
; CHECK-NEXT: [[h:%[0-9]+]] = OpAccessChain [[buffer_ptr]] [[array_buffer_var]] [[one]]
; CHECK-NEXT: [[handle:%[0-9]+]] = OpCopyObject [[buffer_ptr]] [[h]]
  %handle = tail call target("spirv.VulkanBuffer", [0 x %struct.S], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 10, i32 1, i1 false)

; CHECK-NEXT: [[ptr:%[0-9]+]] = OpAccessChain [[storagebuffer_S_ptr]] [[handle]] [[zero]] [[one]]
  %0 = tail call noundef nonnull align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 0) %handle, i32 1)

; CHECK-NEXT: [[ld:%[0-9]+]] = OpLoad [[S_explicit]] [[ptr]] Aligned 4
; CHECK-NEXT: [[copy:%[0-9]+]] = OpCopyLogical [[S]] [[ld]]
  %1 = load %struct.S, ptr addrspace(11) %0, align 4

; CHECK-NEXT: OpReturnValue [[copy]]
  ret %struct.S %1

; CHECK-NEXT: OpFunctionEnd
}
