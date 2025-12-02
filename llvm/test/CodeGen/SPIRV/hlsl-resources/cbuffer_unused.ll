; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; Test that unused and partially unused cbuffers are handled correctly.

; CHECK-DAG: OpDecorate %[[PartiallyUsedCBuffer:[0-9]+]] DescriptorSet 0
; CHECK-DAG: OpDecorate %[[PartiallyUsedCBuffer]] Binding 1
; CHECK-DAG: OpDecorate %[[AnotherCBuffer:[0-9]+]] DescriptorSet 0
; CHECK-DAG: OpDecorate %[[AnotherCBuffer]] Binding 2
; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[v4float:[0-9]+]] = OpTypeVector %[[float]] 4
; CHECK-DAG: %[[_cblayout_PartiallyUsedCBuffer:[0-9]+]] = OpTypeStruct %[[float]]
; CHECK-DAG: %[[_cblayout_AnotherCBuffer:[0-9]+]] = OpTypeStruct %[[v4float]]

%__cblayout_UnusedCBuffer = type <{ float }>
%__cblayout_PartiallyUsedCBuffer = type <{ float, i32 }>
%__cblayout_AnotherCBuffer = type <{ <4 x float>, <4 x float> }>

@UnusedCBuffer.cb = local_unnamed_addr global target("spirv.VulkanBuffer", %__cblayout_UnusedCBuffer, 2, 0) poison
@UnusedCBuffer.str = private unnamed_addr constant [14 x i8] c"UnusedCBuffer\00", align 1
@PartiallyUsedCBuffer.cb = local_unnamed_addr global target("spirv.VulkanBuffer", %__cblayout_PartiallyUsedCBuffer, 2, 0) poison
@used_member = external hidden local_unnamed_addr addrspace(12) global float, align 4
@PartiallyUsedCBuffer.str = private unnamed_addr constant [21 x i8] c"PartiallyUsedCBuffer\00", align 1
@AnotherCBuffer.cb = local_unnamed_addr global target("spirv.VulkanBuffer", %__cblayout_AnotherCBuffer, 2, 0) poison
@a = external hidden local_unnamed_addr addrspace(12) global <4 x float>, align 16
@AnotherCBuffer.str = private unnamed_addr constant [15 x i8] c"AnotherCBuffer\00", align 1
@.str = private unnamed_addr constant [7 x i8] c"output\00", align 1


; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none)
define void @main() local_unnamed_addr #1 {
entry:
  %UnusedCBuffer.cb_h.i.i = tail call target("spirv.VulkanBuffer", %__cblayout_UnusedCBuffer, 2, 0) @llvm.spv.resource.handlefromimplicitbinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @UnusedCBuffer.str)
  store target("spirv.VulkanBuffer", %__cblayout_UnusedCBuffer, 2, 0) %UnusedCBuffer.cb_h.i.i, ptr @UnusedCBuffer.cb, align 8

; CHECK: %[[tmp:[0-9]+]] = OpCopyObject {{%[0-9]+}} %[[PartiallyUsedCBuffer]]
; CHECK: %[[used_member_ptr:.+]] = OpAccessChain %{{.+}} %[[tmp]] %{{.+}} %[[uint_0:[0-9]+]]
  %PartiallyUsedCBuffer.cb_h.i.i = tail call target("spirv.VulkanBuffer", %__cblayout_PartiallyUsedCBuffer, 2, 0) @llvm.spv.resource.handlefromimplicitbinding(i32 1, i32 0, i32 1, i32 0, ptr nonnull @PartiallyUsedCBuffer.str)
  store target("spirv.VulkanBuffer", %__cblayout_PartiallyUsedCBuffer, 2, 0) %PartiallyUsedCBuffer.cb_h.i.i, ptr @PartiallyUsedCBuffer.cb, align 8

; CHECK: %[[tmp:[0-9]+]] = OpCopyObject {{%[0-9]+}} %[[AnotherCBuffer]]
; CHECK: %[[a_ptr:.+]] = OpAccessChain %{{.+}} %[[tmp]] %{{.+}} %[[uint_0]]
  %AnotherCBuffer.cb_h.i.i = tail call target("spirv.VulkanBuffer", %__cblayout_AnotherCBuffer, 2, 0) @llvm.spv.resource.handlefromimplicitbinding(i32 2, i32 0, i32 1, i32 0, ptr nonnull @AnotherCBuffer.str)
  store target("spirv.VulkanBuffer", %__cblayout_AnotherCBuffer, 2, 0) %AnotherCBuffer.cb_h.i.i, ptr @AnotherCBuffer.cb, align 8
  %0 = tail call target("spirv.Image", float, 5, 2, 0, 0, 2, 1) @llvm.spv.resource.handlefromimplicitbinding.tspirv.Image_f32_5_2_0_0_2_1t(i32 3, i32 0, i32 1, i32 0, ptr nonnull @.str)

  %2 = load float, ptr addrspace(12) @used_member, align 4
  %3 = load <4 x float>, ptr addrspace(12) @a, align 16
  %4 = extractelement <4 x float> %3, i64 0
  %add.i = fadd reassoc nnan ninf nsz arcp afn float %4, %2
  %vecinit3.i = insertelement <4 x float> <float poison, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %add.i, i64 0
  %5 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_1t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1) %0, i32 0)
  store <4 x float> %vecinit3.i, ptr addrspace(11) %5, align 16
  ret void
}


attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!hlsl.cbs = !{!0, !1, !2}

!0 = distinct !{ptr @UnusedCBuffer.cb, null}
!1 = distinct !{ptr @PartiallyUsedCBuffer.cb, ptr addrspace(12) @used_member, null}
!2 = distinct !{ptr @AnotherCBuffer.cb, ptr addrspace(12) @a, null}
