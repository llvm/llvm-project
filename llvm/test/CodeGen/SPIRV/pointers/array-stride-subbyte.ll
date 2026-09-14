; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library -filetype=obj < %s | spirv-val --target-env vulkan1.3 %}

; Sub-byte elements must not truncate their ArrayStride to 0.

@.str = private unnamed_addr constant [2 x i8] c"B\00", align 1

; CHECK-DAG: OpDecorate [[i1_array:%[0-9]+]] ArrayStride 1
; CHECK-DAG: OpDecorate [[i4_array:%[0-9]+]] ArrayStride 1
; CHECK-DAG: [[i1_ty:%[0-9]+]] = OpTypeBool
; CHECK-DAG: [[i1_array]] = OpTypeRuntimeArray [[i1_ty]]
; CHECK-DAG: [[i4_ty:%[0-9]+]] = OpTypeInt 8
; CHECK-DAG: [[i4_array]] = OpTypeRuntimeArray [[i4_ty]]

define void @main() local_unnamed_addr #0 {
  %handle1 = tail call target("spirv.VulkanBuffer", [0 x i1], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %handle2 = tail call target("spirv.VulkanBuffer", [0 x i4], 12, 1) @llvm.spv.resource.handlefrombinding(i32 1, i32 0, i32 1, i32 0, ptr nonnull @.str)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
