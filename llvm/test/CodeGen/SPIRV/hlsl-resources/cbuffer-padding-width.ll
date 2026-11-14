; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s --implicit-check-not='"spirv.Padding"'
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; cbuffer padding is modeled with the widest of i32/i16/i8 that divides its
; size, not always i8. The checks are on the struct members, not on module-wide
; Int8: resource name strings emit i8 arrays of their own. Padding shares its
; integer type with real members of the same width and must not be named.

; CHECK-DAG: %[[U8:[0-9]+]] = OpTypeInt 8 0
; CHECK-DAG: %[[U16:[0-9]+]] = OpTypeInt 16 0
; CHECK-DAG: %[[U32:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[HALF:[0-9]+]] = OpTypeFloat 16
; CHECK-DAG: %[[FLOAT:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[VEC4:[0-9]+]] = OpTypeVector %[[FLOAT]] 4

; 12 bytes: i32.
; CHECK-DAG: %[[A:[0-9]+]] = OpTypeStruct %[[FLOAT]] %[[U32]] %[[VEC4]]
; CHECK-DAG: OpMemberDecorate %[[A]] 1 Offset 4
; CHECK-DAG: OpMemberDecorate %[[A]] 2 Offset 16
; 14 bytes: i16.
; CHECK-DAG: %[[B:[0-9]+]] = OpTypeStruct %[[HALF]] %[[U16]] %[[VEC4]]
; CHECK-DAG: OpMemberDecorate %[[B]] 1 Offset 2
; CHECK-DAG: OpMemberDecorate %[[B]] 2 Offset 16
; 3 bytes (odd): i8, shared with the real i8 member.
; CHECK-DAG: %[[C:[0-9]+]] = OpTypeStruct %[[U8]] %[[U8]] %[[FLOAT]]
; CHECK-DAG: OpMemberDecorate %[[C]] 1 Offset 1
; CHECK-DAG: OpMemberDecorate %[[C]] 2 Offset 4
; 4 bytes between real i32 members: one shared i32 type.
; CHECK-DAG: %[[D:[0-9]+]] = OpTypeStruct %[[U32]] %[[U32]] %[[U32]]
; CHECK-DAG: OpMemberDecorate %[[D]] 1 Offset 4
; CHECK-DAG: OpMemberDecorate %[[D]] 2 Offset 8

%A = type <{ float, target("spirv.Padding", 12), <4 x float> }>
%B = type <{ half, target("spirv.Padding", 14), <4 x float> }>
%C = type <{ i8, target("spirv.Padding", 3), float }>
%D = type <{ i32, target("spirv.Padding", 4), i32 }>

@.str.a = private unnamed_addr constant [2 x i8] c"A\00", align 1
@.str.b = private unnamed_addr constant [2 x i8] c"B\00", align 1
@.str.c = private unnamed_addr constant [2 x i8] c"C\00", align 1
@.str.d = private unnamed_addr constant [2 x i8] c"D\00", align 1

define void @main() local_unnamed_addr #0 {
  %ha = tail call target("spirv.VulkanBuffer", %A, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_As_2_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str.a)
  %ba = call ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_As_2_0t(target("spirv.VulkanBuffer", %A, 2, 0) %ha)
  %pa = getelementptr inbounds nuw i8, ptr addrspace(12) %ba, i64 16
  %va = load volatile <4 x float>, ptr addrspace(12) %pa, align 4

  %hb = tail call target("spirv.VulkanBuffer", %B, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_Bs_2_0t(i32 1, i32 0, i32 1, i32 0, ptr nonnull @.str.b)
  %bb = call ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Bs_2_0t(target("spirv.VulkanBuffer", %B, 2, 0) %hb)
  %pb = getelementptr inbounds nuw i8, ptr addrspace(12) %bb, i64 16
  %vb = load volatile <4 x float>, ptr addrspace(12) %pb, align 4

  %hc = tail call target("spirv.VulkanBuffer", %C, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_Cs_2_0t(i32 2, i32 0, i32 1, i32 0, ptr nonnull @.str.c)
  %bc = call ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Cs_2_0t(target("spirv.VulkanBuffer", %C, 2, 0) %hc)
  %pc = getelementptr inbounds nuw i8, ptr addrspace(12) %bc, i64 4
  %vc = load volatile float, ptr addrspace(12) %pc, align 4

  %hd = tail call target("spirv.VulkanBuffer", %D, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_Ds_2_0t(i32 3, i32 0, i32 1, i32 0, ptr nonnull @.str.d)
  %bd = call ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Ds_2_0t(target("spirv.VulkanBuffer", %D, 2, 0) %hd)
  %pd = getelementptr inbounds nuw i8, ptr addrspace(12) %bd, i64 8
  %vd = load volatile i32, ptr addrspace(12) %pd, align 4
  ret void
}

declare target("spirv.VulkanBuffer", %A, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_As_2_0t(i32, i32, i32, i32, ptr)
declare ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_As_2_0t(target("spirv.VulkanBuffer", %A, 2, 0))
declare target("spirv.VulkanBuffer", %B, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_Bs_2_0t(i32, i32, i32, i32, ptr)
declare ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Bs_2_0t(target("spirv.VulkanBuffer", %B, 2, 0))
declare target("spirv.VulkanBuffer", %C, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_Cs_2_0t(i32, i32, i32, i32, ptr)
declare ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Cs_2_0t(target("spirv.VulkanBuffer", %C, 2, 0))
declare target("spirv.VulkanBuffer", %D, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_Ds_2_0t(i32, i32, i32, i32, ptr)
declare ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Ds_2_0t(target("spirv.VulkanBuffer", %D, 2, 0))

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }