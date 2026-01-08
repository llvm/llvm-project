; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.9-library %s | FileCheck %s

define hidden noundef nofpclass(nan inf) float @_Z11test_scalarj(i32 noundef %p0) local_unnamed_addr #0 {
entry:
  ; CHECK : [[UINT:%.*]] = call float @dx.op.legacyF16ToF32(i32 131, i32 %p0)
  ; CHECK : ret float [[UINT]]
  %hlsl.f16tof32 = tail call reassoc nnan ninf nsz arcp afn float @llvm.dx.legacyf16tof32.i32(i32 %p0)
  ret float %hlsl.f16tof32
}

define hidden noundef nofpclass(nan inf) <2 x float> @_Z10test_uint2Dv2_j(<2 x i32> noundef %p0) local_unnamed_addr #0 {
entry:
  ; CHECK: [[UINT2_0:%.*]] = extractelement <2 x i32> %p0, i64 0
  ; CHECK: [[FLOAT_0:%.*]] = call float @dx.op.legacyF16ToF32(i32 131, i32 [[UINT2_0]])
  ; CHECK: [[UINT2_1:%.*]] = extractelement <2 x i32> %p0, i64 1
  ; CHECK: [[FLOAT_1:%.*]] = call float @dx.op.legacyF16ToF32(i32 131, i32 [[UINT2_1]])
  ; CHECK: [[FLOAT2_0:%.*]] = insertelement <2 x float> poison, float [[FLOAT_0]], i64 0
  ; CHECK: [[FLOAT2_1:%.*]] = insertelement <2 x float> [[FLOAT2_0]], float [[FLOAT_1]], i64 1
  ; CHECK : ret <2 x float>  [[FLOAT2_1]]
  %hlsl.f16tof32 = tail call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.dx.legacyf16tof32.v2i32(<2 x i32> %p0)
  ret <2 x float> %hlsl.f16tof32
}

define hidden noundef nofpclass(nan inf) <3 x float> @_Z10test_uint3Dv3_j(<3 x i32> noundef %p0) local_unnamed_addr #0 {
entry:
  ; CHECK: [[UINT3_0:%.*]] = extractelement <3 x i32> %p0, i64 0
  ; CHECK: [[FLOAT_0:%.*]] = call float @dx.op.legacyF16ToF32(i32 131, i32 [[UINT3_0]])
  ; CHECK: [[UINT3_1:%.*]] = extractelement <3 x i32> %p0, i64 1
  ; CHECK: [[FLOAT_1:%.*]] = call float @dx.op.legacyF16ToF32(i32 131, i32 [[UINT3_1]])
  ; CHECK: [[UINT3_2:%.*]] = extractelement <3 x i32> %p0, i64 2
  ; CHECK: [[FLOAT_2:%.*]] = call float @dx.op.legacyF16ToF32(i32 131, i32 [[UINT3_2]])
  ; CHECK: [[FLOAT3_0:%.*]] = insertelement <3 x float> poison, float [[FLOAT_0]], i64 0
  ; CHECK: [[FLOAT3_1:%.*]] = insertelement <3 x float> [[FLOAT3_0]], float [[FLOAT_1]], i64 1
  ; CHECK: [[FLOAT3_2:%.*]] = insertelement <3 x float> [[FLOAT3_1]], float [[FLOAT_2]], i64 2
  ; CHECK : ret <3 x float> [[FLOAT3_2]]
  %hlsl.f16tof32 = tail call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.dx.legacyf16tof32.v3i32(<3 x i32> %p0)
  ret <3 x float> %hlsl.f16tof32
}

define hidden noundef nofpclass(nan inf) <4 x float> @_Z10test_uint4Dv4_j(<4 x i32> noundef %p0) local_unnamed_addr #0 {
entry:
  ; CHECK: [[UINT4_0:%.*]] = extractelement <4 x i32> %p0, i64 0
  ; CHECK: [[FLOAT_0:%.*]] = call float @dx.op.legacyF16ToF32(i32 131, i32 [[UINT4_0]])
  ; CHECK: [[UINT4_1:%.*]] = extractelement <4 x i32> %p0, i64 1
  ; CHECK: [[FLOAT_1:%.*]] = call float @dx.op.legacyF16ToF32(i32 131, i32 [[UINT4_1]])
  ; CHECK: [[UINT4_2:%.*]] = extractelement <4 x i32> %p0, i64 2
  ; CHECK: [[FLOAT_2:%.*]] = call float @dx.op.legacyF16ToF32(i32 131, i32 [[UINT4_2]])
  ; CHECK: [[UINT4_3:%.*]] = extractelement <4 x i32> %p0, i64 3
  ; CHECK: [[FLOAT_3:%.*]] = call float @dx.op.legacyF16ToF32(i32 131, i32 [[UINT4_3]])
  ; CHECK: [[FLOAT4_0:%.*]] = insertelement <4 x float> poison, float [[FLOAT_0]], i64 0
  ; CHECK: [[FLOAT4_1:%.*]] = insertelement <4 x float> [[FLOAT4_0]], float [[FLOAT_1]], i64 1
  ; CHECK: [[FLOAT4_2:%.*]] = insertelement <4 x float> [[FLOAT4_1]], float [[FLOAT_2]], i64 2
  ; CHECK: [[FLOAT4_3:%.*]] = insertelement <4 x float> [[FLOAT4_2]], float [[FLOAT_3]], i64 3
  ; CHECK : ret <4 x float> [[FLOAT4_3]]
  %hlsl.f16tof32 = tail call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.dx.legacyf16tof32.v4i32(<4 x i32> %p0)
  ret <4 x float> %hlsl.f16tof32
}
