; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s

; CHECK-DAG: OpDecorate %[[MyCBuffer:[0-9]+]] DescriptorSet 0
; CHECK-DAG: OpDecorate %[[MyCBuffer]] Binding 0
; CHECK-DAG: OpMemberDecorate %[[__cblayout_MyCBuffer:[0-9]+]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[__cblayout_MyCBuffer]] 1 Offset 16
; CHECK-DAG: %[[uint:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[uint_0:[0-9]+]] = OpConstant %[[uint]] 0{{$}}
; CHECK-DAG: %[[uint_1:[0-9]+]] = OpConstant %[[uint]] 1{{$}}
; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[v4float:[0-9]+]] = OpTypeVector %[[float]] 4
; CHECK-DAG: %[[__cblayout_MyCBuffer]] = OpTypeStruct %[[v4float]] %[[v4float]]
; CHECK-DAG: %[[wrapper:[0-9]+]] = OpTypeStruct %[[__cblayout_MyCBuffer]]
; CHECK-DAG: %[[wrapper_ptr_t:[0-9]+]] = OpTypePointer Uniform %[[wrapper]]
; CHECK-DAG: %[[MyCBuffer]] = OpVariable %[[wrapper_ptr_t]] Uniform
; CHECK-DAG: %[[_ptr_Uniform_v4float:[0-9]+]] = OpTypePointer Uniform %[[v4float]]

%__cblayout_MyCBuffer = type <{ <4 x float>, <4 x float> }>

@MyCBuffer.cb = local_unnamed_addr global target("spirv.VulkanBuffer", target("spirv.Layout", %__cblayout_MyCBuffer, 32, 0, 16), 2, 0) poison
@a = external hidden local_unnamed_addr addrspace(12) global <4 x float>, align 16
@b = external hidden local_unnamed_addr addrspace(12) global <4 x float>, align 16
@MyCBuffer.str = private unnamed_addr constant [10 x i8] c"MyCBuffer\00", align 1
@.str = private unnamed_addr constant [7 x i8] c"output\00", align 1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none)
define void @main() local_unnamed_addr #1 {
entry:
; CHECK: %[[tmp:[0-9]+]] = OpCopyObject %[[wrapper_ptr_t]] %[[MyCBuffer]]
  %MyCBuffer.cb_h.i.i = tail call target("spirv.VulkanBuffer", target("spirv.Layout", %__cblayout_MyCBuffer, 32, 0, 16), 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_tspirv.Layout_s___cblayout_MyCBuffers_32_0_16t_2_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @MyCBuffer.str)
  store target("spirv.VulkanBuffer", target("spirv.Layout", %__cblayout_MyCBuffer, 32, 0, 16), 2, 0) %MyCBuffer.cb_h.i.i, ptr @MyCBuffer.cb, align 8
  %0 = tail call target("spirv.Image", <4 x float>, 5, 2, 0, 0, 2, 3) @llvm.spv.resource.handlefrombinding.tspirv.Image_v4f32_5_2_0_0_2_3t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
; CHECK: %[[a_ptr:.+]] = OpAccessChain %[[_ptr_Uniform_v4float]] %[[tmp]] %[[uint_0]] %[[uint_0]]
; CHECK: %[[b_ptr:.+]] = OpAccessChain %[[_ptr_Uniform_v4float]] %[[tmp]] %[[uint_0]] %[[uint_1]]
; CHECK: %[[a_val:.+]] = OpLoad %[[v4float]] %[[a_ptr]]
; CHECK: %[[b_val:.+]] = OpLoad %[[v4float]] %[[b_ptr]]
  %a_val = load <4 x float>, ptr addrspace(12) @a, align 16
  %b_val = load <4 x float>, ptr addrspace(12) @b, align 16
  %add = fadd <4 x float> %a_val, %b_val
  %output_ptr = tail call noundef ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_v4f32_5_2_0_0_2_3t(target("spirv.Image", <4 x float>, 5, 2, 0, 0, 2, 3) %0, i32 0)
  store <4 x float> %add, ptr addrspace(11) %output_ptr, align 16
  ret void
}

attributes #1 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!hlsl.cbs = !{!0}

!0 = !{ptr @MyCBuffer.cb, ptr addrspace(12) @a, ptr addrspace(12) @b}
