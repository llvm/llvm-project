; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; Test that uses of cbuffer members are handled correctly.

; CHECK-DAG: OpDecorate %[[MyCBuffer:[0-9]+]] DescriptorSet 0
; CHECK-DAG: OpDecorate %[[MyCBuffer]] Binding 0
; CHECK-DAG: OpMemberDecorate %[[__cblayout_MyCBuffer:[0-9]+]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[__cblayout_MyCBuffer]] 1 Offset 16
; CHECK-DAG: %[[uint:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[uint_0:[0-9]+]] = OpConstant %[[uint]] 0{{$}}
; CHECK-DAG: %[[uint_1:[0-9]+]] = OpConstant %[[uint]] 1{{$}}
; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[v4float:[0-9]+]] = OpTypeVector %[[float]] 4
; CHECK-DAG: %[[MyStruct:[0-9]+]] = OpTypeStruct %[[v4float]]
; CHECK-DAG: %[[__cblayout_MyCBuffer]] = OpTypeStruct %[[MyStruct]] %[[v4float]]
; CHECK-DAG: %[[wrapper:[0-9]+]] = OpTypeStruct %[[__cblayout_MyCBuffer]]
; CHECK-DAG: %[[wrapper_ptr_t:[0-9]+]] = OpTypePointer Uniform %[[wrapper]]
; CHECK-DAG: %[[MyCBuffer]] = OpVariable %[[wrapper_ptr_t]] Uniform
; CHECK-DAG: %[[_ptr_Uniform_v4float:[0-9]+]] = OpTypePointer Uniform %[[v4float]]
; CHECK-DAG: %[[_ptr_Uniform_float:[0-9]+]] = OpTypePointer Uniform %[[float]]

%MyStruct = type { <4 x float> }
%__cblayout_MyCBuffer = type <{ %MyStruct, <4 x float> }>

@MyCBuffer.cb = local_unnamed_addr global target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) poison
@s = external hidden local_unnamed_addr addrspace(12) global %MyStruct, align 16
@v = external hidden local_unnamed_addr addrspace(12) global <4 x float>, align 16
@MyCBuffer.str = private unnamed_addr constant [10 x i8] c"MyCBuffer\00", align 1
@.str = private unnamed_addr constant [7 x i8] c"output\00", align 1

define void @main() {
entry:
; CHECK: %[[tmp:[0-9]+]] = OpCopyObject %[[wrapper_ptr_t]] %[[MyCBuffer]]
  %MyCBuffer.cb_h.i.i = tail call target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @MyCBuffer.str)
  store target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) %MyCBuffer.cb_h.i.i, ptr @MyCBuffer.cb, align 8
  %0 = tail call target("spirv.Image", float, 5, 2, 0, 0, 2, 3) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_3t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)

; CHECK: %[[tmp_ptr:[0-9]+]] = OpAccessChain {{%[0-9]+}} %[[tmp]] %[[uint_0]] %[[uint_0]]
; CHECK: %[[v_ptr:.+]] = OpAccessChain %[[_ptr_Uniform_v4float]] %[[tmp]] %[[uint_0]] %[[uint_1]]
; CHECK: %[[s_ptr_gep:[0-9]+]] = OpInBoundsAccessChain %[[_ptr_Uniform_float]] %[[tmp_ptr]] %[[uint_0]] %[[uint_1]]
; CHECK: %[[s_val:.+]] = OpLoad %[[float]] %[[s_ptr_gep]]
  %load_from_gep = load float, ptr addrspace(12) getelementptr inbounds (%MyStruct, ptr addrspace(12) @s, i32 0, i32 0, i32 1), align 4

; CHECK: %[[v_val:.+]] = OpLoad %[[v4float]] %[[v_ptr]]
  %load_v = load <4 x float>, ptr addrspace(12) @v, align 16

  %extract_v = extractelement <4 x float> %load_v, i64 0
  %add = fadd float %load_from_gep, %extract_v
  %get_output_ptr = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_3t(target("spirv.Image", float, 5, 2, 0, 0, 2, 3) %0, i32 0)
  store float %add, ptr addrspace(11) %get_output_ptr, align 4
  ret void
}

!hlsl.cbs = !{!0}
!0 = !{ptr @MyCBuffer.cb, ptr addrspace(12) @s, ptr addrspace(12) @v}
