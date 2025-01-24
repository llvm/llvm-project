; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("spirv.VulkanBuffer", [0 x i32], 12, 0, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_0_0t(i32, i32, i32, i32, i1) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_i32_5_2_0_0_2_0t(i32, i32, i32, i32, i1) #0

; CHECK-DAG: OpDecorate [[BufferVar:%.+]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[BufferVar]] Binding 0
; CHECK-DAG: OpDecorate [[BufferType:%.+]] Block
; STVEN-CHECK-DAG: OpMemberDecorate [[BufferType]] 0 Offset 0 // The decoration is not output correctly in the assembly output
; CHECK-DAG: OpMemberDecorate [[BufferType]] 0 NonWritable
; CHECK-DAG: OpDecorate [[ImageVar:%.+]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[ImageVar]] Binding 1


; CHECK-DAG:  [[ArrayType:%.+]] = OpTypeRuntimeArray
; CHECK-DAG:  [[BufferType]] = OpTypeStruct [[ArrayType]]
; CHECK-DAG: [[BufferPtrType:%.+]] = OpTypePointer StorageBuffer [[BufferType]]
; CHECK-DAG: [[int:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[zero:%[0-9]+]] = OpConstant [[int]] 0
; CHECK-DAG: [[one:%[0-9]+]] = OpConstant [[int]] 1
; CHECK-DAG: [[BufferVar]] = OpVariable [[BufferPtrType]] StorageBuffer
; CHECK-DAG: [[ImageVar]] = OpVariable {{.*}} UniformConstant

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define void @main() local_unnamed_addr #1 {
entry:
; CHECK: [[BufferHandle:%.+]] = OpCopyObject [[BufferPtrType]] [[BufferVar]]
  %i_h.i.i = tail call target("spirv.VulkanBuffer", [0 x i32], 12, 0, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false)

  %o_h.i.i = tail call target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_i32_5_2_0_0_2_0t(i32 0, i32 1, i32 1, i32 0, i1 false)

; CHECK: [[AC:%.+]] = OpAccessChain {{.*}} [[BufferHandle]] [[zero]] [[one]]
  %0 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_0_0t(target("spirv.VulkanBuffer", [0 x i32], 12, 0, 0) %i_h.i.i, i32 1)

; CHECK: [[LD:%.+]] = OpLoad [[int]] [[AC]] Aligned 4
  %1 = load i32, ptr addrspace(11) %0, align 4, !tbaa !3

; CHECK: [[ImageHandle:%.+]] = OpLoad {{.*}} [[ImageVar]]
  %2 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %o_h.i.i, i32 0)

; CHECK: OpImageWrite [[ImageHandle]] [[zero]] [[LD]]
  store i32 %1, ptr addrspace(11) %2, align 4, !tbaa !3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0), i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_0_0t(target("spirv.VulkanBuffer", [0 x i32], 12, 0, 0), i32) #0

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "approx-func-fp-math"="false" "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 21.0.0git (git@github.com:s-perron/llvm-project.git 6e86add06c03e328dbb4b83f99406cc832a22f86)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
