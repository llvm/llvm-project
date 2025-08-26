; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-unknown-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G10"

@.str.b0 = private unnamed_addr constant [3 x i8] c"B0\00", align 1
@.str.b1 = private unnamed_addr constant [3 x i8] c"B1\00", align 1
@.str.b2 = private unnamed_addr constant [3 x i8] c"B2\00", align 1
@.str.b3 = private unnamed_addr constant [3 x i8] c"B3\00", align 1
@.str.b4 = private unnamed_addr constant [3 x i8] c"B4\00", align 1

; CHECK-DAG: OpName [[standard_layout:%[0-9]+]] "standard_layout"
; CHECK-DAG: OpMemberDecorate [[standard_layout]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate [[standard_layout]] 1 Offset 4

; CHECK-DAG: OpName [[standard_layout_with_different_offset:%[0-9]+]] "standard_layout"
; CHECK-DAG: OpMemberDecorate [[standard_layout_with_different_offset]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate [[standard_layout_with_different_offset]] 1 Offset 8
%standard_layout = type { i32, i32 }

; CHECK-DAG: OpName [[backwards_layout:%[0-9]+]] "backwards_layout"
; CHECK-DAG: OpMemberDecorate [[backwards_layout]] 0 Offset 4
; CHECK-DAG: OpMemberDecorate [[backwards_layout]] 1 Offset 0
%backwards_layout = type { i32, i32 }

; CHECK-DAG: OpName [[large_gap:%[0-9]+]] "large_gap"
; CHECK-DAG: OpMemberDecorate [[large_gap]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate [[large_gap]] 1 Offset 64
; CHECK-DAG: OpMemberDecorate [[large_gap]] 2 Offset 1020
; CHECK-DAG: OpMemberDecorate [[large_gap]] 3 Offset 4
%large_gap = type { i32, i32, i32, i32 }

; CHECK-DAG: OpName [[mixed_layout:%[0-9]+]] "mixed_layout"
; CHECK-DAG: OpMemberDecorate [[mixed_layout]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate [[mixed_layout]] 1 Offset 8
; CHECK-DAG: OpMemberDecorate [[mixed_layout]] 2 Offset 4
; CHECK-DAG: OpMemberDecorate [[mixed_layout]] 3 Offset 12
%mixed_layout = type { i32, i32, i32, i32 }

define void @main() local_unnamed_addr #1 {
entry:
  %standard_handle = tail call target("spirv.VulkanBuffer", target("spirv.Layout", %standard_layout, 8, 0, 4), 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_tspirv.Layout_s___cblayout_Bs_8_0_4t_2_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.b0)
  %standard_handle_with_different_offset = tail call target("spirv.VulkanBuffer", target("spirv.Layout", %standard_layout, 12, 0, 8), 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_tspirv.Layout_s___cblayout_Bs_8_0_4t_2_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.b1)
  %backwards_handle = tail call target("spirv.VulkanBuffer", target("spirv.Layout", %backwards_layout, 8, 4, 0), 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_tspirv.Layout_s___cblayout_Bs_8_0_4t_2_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.b2)
  %large_gap_handle = tail call target("spirv.VulkanBuffer", target("spirv.Layout", %large_gap, 1024, 0, 64, 1020, 4), 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_tspirv.Layout_s___cblayout_Bs_8_0_4t_2_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.b3)
  %mixed_handle = tail call target("spirv.VulkanBuffer", target("spirv.Layout", %mixed_layout, 16, 0, 8, 4, 12), 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_tspirv.Layout_s___cblayout_Bs_8_0_4t_2_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.b4)
  ret void
}

attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) "approx-func-fp-math"="false" "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
