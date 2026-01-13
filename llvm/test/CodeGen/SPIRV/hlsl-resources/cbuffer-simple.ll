; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[FLOAT:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[VEC4:[0-9]+]] = OpTypeVector %[[FLOAT]] 4
; CHECK-DAG: %[[PTR_FLOAT:[0-9]+]] = OpTypePointer Uniform %[[FLOAT]]
; CHECK-DAG: %[[PTR_VEC4:[0-9]+]] = OpTypePointer Uniform %[[VEC4]]
; CHECK-DAG: %[[STRUCT:[0-9]+]] = OpTypeStruct %[[VEC4]] %[[FLOAT]]
; CHECK-DAG: %[[CBUFFER_TYPE:[0-9]+]] = OpTypeStruct %[[STRUCT]]
; CHECK-DAG: %[[PTR_CBUFFER:[0-9]+]] = OpTypePointer Uniform %[[CBUFFER_TYPE]]
; CHECK-DAG: %[[INT:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[ZERO:[0-9]+]] = OpConstant %[[INT]] 0{{$}}
; CHECK-DAG: %[[ONE:[0-9]+]] = OpConstant %[[INT]] 1{{$}}

; CHECK-DAG: OpMemberDecorate %[[STRUCT]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[STRUCT]] 1 Offset 16
; CHECK-DAG: OpMemberDecorate %[[CBUFFER_TYPE]] 0 Offset 0
; CHECK-DAG: OpDecorate %[[CBUFFER_TYPE]] Block

; CHECK-DAG: %[[CBUFFER:[0-9]+]] = OpVariable %[[PTR_CBUFFER]] Uniform

%__cblayout_MyCBuffer = type <{ <4 x float>, float }>

@MyCBuffer.cb = local_unnamed_addr global target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) poison
@color = external hidden local_unnamed_addr addrspace(12) global <4 x float>, align 16
@factor = external hidden local_unnamed_addr addrspace(12) global float, align 4
@MyCBuffer.str = private unnamed_addr constant [10 x i8] c"MyCBuffer\00", align 1
@.str = private unnamed_addr constant [7 x i8] c"output\00", align 1

declare target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s___cblayout_MyCBuffers_2_0t(i32, i32, i32, i32, ptr)

define void @main() #1 {
entry:
; CHECK: %[[COPY:[0-9]+]] = OpCopyObject %[[PTR_CBUFFER]] %[[CBUFFER]]
; CHECK: %[[PTR_VEC4_ACCESS:[0-9]+]] = OpAccessChain %[[PTR_VEC4]] %[[COPY]] %[[ZERO]] %[[ZERO]]
; CHECK: %[[PTR_FLOAT_ACCESS:[0-9]+]] = OpAccessChain %[[PTR_FLOAT]] %[[COPY]] %[[ZERO]] %[[ONE]]
  %MyCBuffer.cb_h.i.i = tail call target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s___cblayout_MyCBuffers_2_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @MyCBuffer.str)
  store target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) %MyCBuffer.cb_h.i.i, ptr @MyCBuffer.cb, align 8

  %0 = tail call target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %1 = tail call i32 @llvm.spv.thread.id.i32(i32 0)
  %2 = tail call i32 @llvm.spv.thread.id.i32(i32 1)
  %conv.i = uitofp i32 %1 to float
  %conv2.i = uitofp i32 %2 to float
  %3 = insertelement <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, float %conv.i, i64 0
  %vecinit5.i = insertelement <4 x float> %3, float %conv2.i, i64 1

; CHECK: %[[VAL_VEC4:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_VEC4_ACCESS]] Aligned 16
  %4 = load <4 x float>, ptr addrspace(12) @color, align 16
  %mul.i = fmul reassoc nnan ninf nsz arcp afn <4 x float> %vecinit5.i, %4

; CHECK: %[[VAL_FLOAT:[0-9]+]] = OpLoad %[[FLOAT]] %[[PTR_FLOAT_ACCESS]] Aligned 4
  %5 = load float, ptr addrspace(12) @factor, align 4

  %splat.splatinsert.i = insertelement <4 x float> poison, float %5, i64 0
  %splat.splat.i = shufflevector <4 x float> %splat.splatinsert.i, <4 x float> poison, <4 x i32> zeroinitializer
  %mul6.i = fmul reassoc nnan ninf nsz arcp afn <4 x float> %mul.i, %splat.splat.i
  %6 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) %0, i32 0)
  store <4 x float> %mul6.i, ptr addrspace(11) %6, align 16
  ret void
}

declare i32 @llvm.spv.thread.id.i32(i32)

declare target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32, i32, i32, i32, ptr)

declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1), i32)

attributes #1 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!hlsl.cbs = !{!0}

!0 = !{ptr @MyCBuffer.cb, ptr addrspace(12) @color, ptr addrspace(12) @factor}
