; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s --implicit-check-not=Aliased
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -filetype=obj -o - | spirv-val --target-env vulkan1.3 %}

; Resources sharing a set and binding should be given the "Aliased" decoration

; CHECK-DAG: OpName %[[#A:]] "a"
; CHECK-DAG: OpName %[[#B:]] "b"
; CHECK-DAG: OpName %[[#RAW_I32:]] ".resource.alias.0.3.i32.rw.raw"
; CHECK-DAG: OpName %[[#RAW_F32:]] ".resource.alias.0.3.float.rw.raw"
; CHECK-DAG: OpDecorate %[[#A]] Aliased
; CHECK-DAG: OpDecorate %[[#B]] Aliased
; CHECK-DAG: OpDecorate %[[#RAW_I32]] Aliased
; CHECK-DAG: OpDecorate %[[#RAW_F32]] Aliased

@a = private unnamed_addr constant [2 x i8] c"a\00"
@b = private unnamed_addr constant [2 x i8] c"b\00"
@c = private unnamed_addr constant [2 x i8] c"c\00"
@raw = private unnamed_addr constant [4 x i8] c"raw\00"

define void @main() #0 {
  %ha = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @a)
  %hb = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @b)
  %hc = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr @c)
  %pa = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %ha, i32 0)
  %pb = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x float], 12, 1) %hb, i32 0)
  %pc = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %hc, i32 0)
  %va = load i32, ptr addrspace(11) %pa
  store i32 %va, ptr addrspace(11) %pc
  %vb = load float, ptr addrspace(11) %pb
  store float %vb, ptr addrspace(11) %pb

  %hraw = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, ptr @raw)
  %i = call i32 @llvm.spv.resource.load.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %hraw, i32 0, i32 4)
  %f = call float @llvm.spv.resource.load.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %hraw, i32 4, i32 4)
  call void @llvm.spv.resource.store.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %hraw, i32 8, i32 %i, i32 4)
  call void @llvm.spv.resource.store.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %hraw, i32 12, float %f, i32 4)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
