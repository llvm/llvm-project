; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[FLOAT:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[VEC4:[0-9]+]] = OpTypeVector %[[FLOAT]] 4
; CHECK-DAG: %[[PTR_VEC4:[0-9]+]] = OpTypePointer Uniform %[[VEC4]]
; CHECK-DAG: %[[INT:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[ZERO:[0-9]+]] = OpConstant %[[INT]] 0{{$}}

; CHECK-DAG: %[[STRUCT_MATRIX:[0-9]+]] = OpTypeStruct %[[VEC4]] %[[VEC4]] %[[VEC4]] %[[VEC4]]
; CHECK-DAG: %[[PTR_MATRIX:[0-9]+]] = OpTypePointer Uniform %[[STRUCT_MATRIX]]
; CHECK-DAG: %[[PTR_FLOAT:[0-9]+]] = OpTypePointer Uniform %[[FLOAT]]

; CHECK-DAG: %[[STRUCT_MYSTRUCT:[0-9]+]] = OpTypeStruct %[[STRUCT_MATRIX]] %[[STRUCT_MATRIX]] %[[STRUCT_MATRIX]]

; CHECK-DAG: %[[PTR_MYSTRUCT:[0-9]+]] = OpTypePointer Uniform %[[STRUCT_MYSTRUCT]]
; CHECK-DAG: %[[STRUCT_INNER:[0-9]+]] = OpTypeStruct %[[STRUCT_MYSTRUCT]] %[[FLOAT]]

; CHECK-DAG: %[[STRUCT_CBUFFER:[0-9]+]] = OpTypeStruct %[[STRUCT_INNER]]
; CHECK-DAG: %[[PTR_CBUFFER:[0-9]+]] = OpTypePointer Uniform %[[STRUCT_CBUFFER]]
; CHECK-DAG: %[[INT64:[0-9]+]] = OpTypeInt 64 0

; CHECK-DAG: OpMemberDecorate %[[STRUCT_CBUFFER]] 0 Offset 0
; CHECK-DAG: OpDecorate %[[STRUCT_CBUFFER]] Block
; CHECK-DAG: OpMemberDecorate %[[STRUCT_INNER]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[STRUCT_INNER]] 1 Offset 192
; CHECK-DAG: OpMemberDecorate %[[STRUCT_MYSTRUCT]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[STRUCT_MYSTRUCT]] 1 Offset 64
; CHECK-DAG: OpMemberDecorate %[[STRUCT_MYSTRUCT]] 2 Offset 128
; CHECK-DAG: OpMemberDecorate %[[STRUCT_MATRIX]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[STRUCT_MATRIX]] 1 Offset 16
; CHECK-DAG: OpMemberDecorate %[[STRUCT_MATRIX]] 2 Offset 32
; CHECK-DAG: OpMemberDecorate %[[STRUCT_MATRIX]] 3 Offset 48

; CHECK-DAG: %[[ONE:[0-9]+]] = OpConstant %[[INT]] 1{{$}}
; CHECK-DAG: %[[ZERO_64:[0-9]+]] = OpConstant %[[INT64]] 0{{$}}
; CHECK-DAG: %[[ONE_64:[0-9]+]] = OpConstant %[[INT64]] 1{{$}}
; CHECK-DAG: %[[TWO_64:[0-9]+]] = OpConstant %[[INT64]] 2{{$}}
; CHECK-DAG: %[[THREE_64:[0-9]+]] = OpConstant %[[INT64]] 3{{$}}

; CHECK: %[[CBUFFER:[0-9]+]] = OpVariable %[[PTR_CBUFFER]] Uniform

%__cblayout_MyCBuffer = type <{ %MyStruct, float }>
%MyStruct = type <{ %MyMatrix, %MyMatrix, %MyMatrix }>
%MyMatrix = type <{ <4 x float>, <4 x float>, <4 x float>, <4 x float> }>

@MyCBuffer.cb = local_unnamed_addr global target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) poison
@transforms = external hidden local_unnamed_addr addrspace(12) global %MyStruct, align 1
@blend = external hidden local_unnamed_addr addrspace(12) global float, align 4
@MyCBuffer.str = private unnamed_addr constant [10 x i8] c"MyCBuffer\00", align 1
@.str = private unnamed_addr constant [7 x i8] c"output\00", align 1

declare target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s___cblayout_MyCBuffers_2_0t(i32, i32, i32, i32, ptr)

declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>)

define void @main() #3 {
entry:
; CHECK: %[[COPY:[0-9]+]] = OpCopyObject %[[PTR_CBUFFER]] %[[CBUFFER]]
; CHECK: %[[PTR_STRUCT:[0-9]+]] = OpAccessChain %[[PTR_MYSTRUCT]] %[[COPY]] %[[ZERO]] %[[ZERO]]
; CHECK: %[[PTR_FLOAT_VAL:[0-9]+]] = OpAccessChain %[[PTR_FLOAT]] %[[COPY]] %[[ZERO]] %[[ONE]]
  %MyCBuffer.cb_h.i.i = tail call target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s___cblayout_MyCBuffers_2_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @MyCBuffer.str)
  store target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) %MyCBuffer.cb_h.i.i, ptr @MyCBuffer.cb, align 8

  %0 = tail call target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %1 = tail call i32 @llvm.spv.thread.id.i32(i32 0)
  %2 = tail call i32 @llvm.spv.thread.id.i32(i32 1)
  %conv.i = uitofp i32 %1 to float
  %conv2.i = uitofp i32 %2 to float
  %3 = insertelement <4 x float> poison, float %conv.i, i64 0

; CHECK: %[[PTR_M0_V0:[0-9]+]] = OpAccessChain %[[PTR_VEC4]] %[[PTR_STRUCT]] %[[ZERO]] %[[ZERO]]
; CHECK: %[[VAL_M0_V0:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_M0_V0]] Aligned 16
  %4 = load <4 x float>, ptr addrspace(12) @transforms, align 16

; CHECK: %[[PTR_M0_V1:[0-9]+]] = OpInBoundsAccessChain %[[PTR_VEC4]] %[[PTR_STRUCT]] %[[ZERO_64]] %[[ONE_64]]
; CHECK: %[[VAL_M0_V1:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_M0_V1]] Aligned 16
  %5 = load <4 x float>, ptr addrspace(12) getelementptr inbounds nuw (i8, ptr addrspace(12) @transforms, i64 16), align 16

; CHECK: %[[PTR_M0_V3:[0-9]+]] = OpInBoundsAccessChain %[[PTR_VEC4]] %[[PTR_STRUCT]] %[[ZERO_64]] %[[THREE_64]]
; CHECK: %[[VAL_M0_V3:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_M0_V3]] Aligned 16
  %6 = load <4 x float>, ptr addrspace(12) getelementptr inbounds nuw (i8, ptr addrspace(12) @transforms, i64 48), align 16

  %splat.splat.i18.i = shufflevector <4 x float> %3, <4 x float> poison, <4 x i32> zeroinitializer
  %7 = insertelement <4 x float> poison, float %conv2.i, i64 0
  %splat.splat2.i19.i = shufflevector <4 x float> %7, <4 x float> poison, <4 x i32> zeroinitializer
  %mul3.i20.i = fmul reassoc nnan ninf nsz arcp afn <4 x float> %splat.splat2.i19.i, %5
  %8 = tail call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.fmuladd.v4f32(<4 x float> %splat.splat.i18.i, <4 x float> nofpclass(nan inf) %4, <4 x float> %mul3.i20.i)
  %9 = fadd reassoc nnan ninf nsz arcp afn <4 x float> %8, %6
; CHECK: %[[PTR_M1:[0-9]+]] = OpInBoundsAccessChain %[[PTR_MATRIX]] %[[PTR_STRUCT]] %[[ONE_64]]
; CHECK: %[[PTR_M1_V0:[0-9]+]] = OpAccessChain %[[PTR_VEC4]] %[[PTR_M1]] %[[ZERO]]
; CHECK: %[[VAL_M1_V0:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_M1_V0]] Aligned 16
  %10 = load <4 x float>, ptr addrspace(12) getelementptr inbounds nuw (i8, ptr addrspace(12) @transforms, i64 64), align 16
; CHECK: %[[PTR_M1_V1:[0-9]+]] = OpInBoundsAccessChain %[[PTR_VEC4]] %[[PTR_STRUCT]] %[[ONE_64]] %[[ONE_64]]
; CHECK: %[[VAL_M1_V1:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_M1_V1]] Aligned 16
  %11 = load <4 x float>, ptr addrspace(12) getelementptr inbounds nuw (i8, ptr addrspace(12) @transforms, i64 80), align 16
; CHECK: %[[PTR_M1_V2:[0-9]+]] = OpInBoundsAccessChain %[[PTR_VEC4]] %[[PTR_STRUCT]] %[[ONE_64]] %[[TWO_64]]
; CHECK: %[[VAL_M1_V2:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_M1_V2]] Aligned 16
  %12 = load <4 x float>, ptr addrspace(12) getelementptr inbounds nuw (i8, ptr addrspace(12) @transforms, i64 96), align 16
; CHECK: %[[PTR_M1_V3:[0-9]+]] = OpInBoundsAccessChain %[[PTR_VEC4]] %[[PTR_STRUCT]] %[[ONE_64]] %[[THREE_64]]
; CHECK: %[[VAL_M1_V3:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_M1_V3]] Aligned 16
  %13 = load <4 x float>, ptr addrspace(12) getelementptr inbounds nuw (i8, ptr addrspace(12) @transforms, i64 112), align 16
  %splat.splat.i13.i = shufflevector <4 x float> %9, <4 x float> poison, <4 x i32> zeroinitializer
  %splat.splat2.i14.i = shufflevector <4 x float> %9, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mul3.i15.i = fmul reassoc nnan ninf nsz arcp afn <4 x float> %splat.splat2.i14.i, %11
  %14 = tail call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.fmuladd.v4f32(<4 x float> %splat.splat.i13.i, <4 x float> nofpclass(nan inf) %10, <4 x float> %mul3.i15.i)
  %splat.splat5.i16.i = shufflevector <4 x float> %9, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %15 = tail call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.fmuladd.v4f32(<4 x float> %splat.splat5.i16.i, <4 x float> nofpclass(nan inf) %12, <4 x float> %14)
  %splat.splat7.i17.i = shufflevector <4 x float> %9, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %16 = tail call reassoc nnan ninf nsz arcp afn noundef <4 x float> @llvm.fmuladd.v4f32(<4 x float> %splat.splat7.i17.i, <4 x float> nofpclass(nan inf) %13, <4 x float> %15)
; CHECK: %[[PTR_M2:[0-9]+]] = OpInBoundsAccessChain %[[PTR_MATRIX]] %[[PTR_STRUCT]] %[[TWO_64]]
; CHECK: %[[PTR_M2_V0:[0-9]+]] = OpAccessChain %[[PTR_VEC4]] %[[PTR_M2]] %[[ZERO]]
; CHECK: %[[VAL_M2_V0:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_M2_V0]] Aligned 16
  %17 = load <4 x float>, ptr addrspace(12) getelementptr inbounds nuw (i8, ptr addrspace(12) @transforms, i64 128), align 16
; CHECK: %[[PTR_M2_V1:[0-9]+]] = OpInBoundsAccessChain %[[PTR_VEC4]] %[[PTR_STRUCT]] %[[TWO_64]] %[[ONE_64]]
; CHECK: %[[VAL_M2_V1:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_M2_V1]] Aligned 16
  %18 = load <4 x float>, ptr addrspace(12) getelementptr inbounds nuw (i8, ptr addrspace(12) @transforms, i64 144), align 16
; CHECK: %[[PTR_M2_V2:[0-9]+]] = OpInBoundsAccessChain %[[PTR_VEC4]] %[[PTR_STRUCT]] %[[TWO_64]] %[[TWO_64]]
; CHECK: %[[VAL_M2_V2:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_M2_V2]] Aligned 16
  %19 = load <4 x float>, ptr addrspace(12) getelementptr inbounds nuw (i8, ptr addrspace(12) @transforms, i64 160), align 16
; CHECK: %[[PTR_M2_V3:[0-9]+]] = OpInBoundsAccessChain %[[PTR_VEC4]] %[[PTR_STRUCT]] %[[TWO_64]] %[[THREE_64]]
; CHECK: %[[VAL_M2_V3:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_M2_V3]] Aligned 16
  %20 = load <4 x float>, ptr addrspace(12) getelementptr inbounds nuw (i8, ptr addrspace(12) @transforms, i64 176), align 16
  %splat.splat.i.i = shufflevector <4 x float> %16, <4 x float> poison, <4 x i32> zeroinitializer
  %splat.splat2.i.i = shufflevector <4 x float> %16, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mul3.i.i = fmul reassoc nnan ninf nsz arcp afn <4 x float> %splat.splat2.i.i, %18
  %21 = tail call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.fmuladd.v4f32(<4 x float> %splat.splat.i.i, <4 x float> nofpclass(nan inf) %17, <4 x float> %mul3.i.i)
  %splat.splat5.i.i = shufflevector <4 x float> %16, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %22 = tail call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.fmuladd.v4f32(<4 x float> %splat.splat5.i.i, <4 x float> nofpclass(nan inf) %19, <4 x float> %21)
  %splat.splat7.i.i = shufflevector <4 x float> %16, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %23 = tail call reassoc nnan ninf nsz arcp afn noundef <4 x float> @llvm.fmuladd.v4f32(<4 x float> %splat.splat7.i.i, <4 x float> nofpclass(nan inf) %20, <4 x float> %22)
  %24 = load float, ptr addrspace(12) @blend, align 4
; CHECK: %[[VAL_FLOAT:[0-9]+]] = OpLoad %[[FLOAT]] %[[PTR_FLOAT_VAL]] Aligned 4
; CHECK: %[[SPLAT_INS:[0-9]+]] = OpCompositeInsert %[[VEC4]] %[[VAL_FLOAT]] {{.*}} 0
; CHECK: %[[SPLAT:[0-9]+]] = OpVectorShuffle %[[VEC4]] %[[SPLAT_INS]] {{.*}} 0 0 0 0
; CHECK: %[[RES:[0-9]+]] = OpFMul %[[VEC4]] {{%[0-9]+}} %[[SPLAT]]
  %splat.splatinsert.i = insertelement <4 x float> poison, float %24, i64 0
  %splat.splat.i = shufflevector <4 x float> %splat.splatinsert.i, <4 x float> poison, <4 x i32> zeroinitializer
  %mul.i = fmul reassoc nnan ninf nsz arcp afn <4 x float> %23, %splat.splat.i
  %25 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) %0, i32 0)
  store <4 x float> %mul.i, ptr addrspace(11) %25, align 16
; CHECK: OpStore {{%[0-9]+}} %[[RES]] Aligned 16
  ret void
}

declare i32 @llvm.spv.thread.id.i32(i32)

declare target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32, i32, i32, i32, ptr)

declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1), i32)

attributes #1 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #3 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #4 = { mustprogress nofree nosync nounwind willreturn memory(none) }

!hlsl.cbs = !{!0}

!0 = !{ptr @MyCBuffer.cb, ptr addrspace(12) @transforms, ptr addrspace(12) @blend}