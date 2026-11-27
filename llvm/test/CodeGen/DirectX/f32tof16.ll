; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.9-library %s | FileCheck %s

define hidden noundef i32 @_Z11test_scalarj(float noundef %p0) local_unnamed_addr #0 {
entry:
  ; CHECK : [[UINT:%.*]] = call i32 @dx.op.legacyF32ToF16(i32 130, float %p0)
  ; CHECK : ret i32 [[UINT]]
  %hlsl.f32tof16 = tail call i32 @llvm.dx.legacyf32tof16.i32(float %p0)
  ret i32 %hlsl.f32tof16
}

define hidden noundef <2 x i32> @_Z10test_uint2Dv2_j(<2 x float> noundef %p0) local_unnamed_addr #0 {
entry:
  ; CHECK: [[FLOAT2_0:%.*]] = extractelement <2 x float> %p0, i64 0
  ; CHECK: [[UINT_0:%.*]] = call i32 @dx.op.legacyF32ToF16(i32 130, float [[FLOAT2_0]])
  ; CHECK: [[FLOAT2_1:%.*]] = extractelement <2 x float> %p0, i64 1
  ; CHECK: [[UINT_1:%.*]] = call i32 @dx.op.legacyF32ToF16(i32 130, float [[FLOAT2_1]])
  ; CHECK: [[UINT2_0:%.*]] = insertelement <2 x i32> poison, i32 [[UINT_0]], i64 0
  ; CHECK: [[UINT2_1:%.*]] = insertelement <2 x i32> [[UINT2_0]], i32 [[UINT_1]], i64 1
  ; CHECK : ret <2 x i32>  [[UINT2_1]]
  %hlsl.f32tof16 = tail call <2 x i32> @llvm.dx.legacyf32tof16.v2i32(<2 x float> %p0)
  ret <2 x i32> %hlsl.f32tof16
}

define hidden noundef <3 x i32> @_Z10test_uint3Dv3_j(<3 x float> noundef %p0) local_unnamed_addr #0 {
entry:
  ; CHECK: [[FLOAT3_0:%.*]] = extractelement <3 x float> %p0, i64 0
  ; CHECK: [[UINT_0:%.*]] = call i32 @dx.op.legacyF32ToF16(i32 130, float [[FLOAT3_0]])
  ; CHECK: [[FLOAT3_1:%.*]] = extractelement <3 x float> %p0, i64 1
  ; CHECK: [[UINT_1:%.*]] = call i32 @dx.op.legacyF32ToF16(i32 130, float [[FLOAT3_1]])
  ; CHECK: [[FLOAT3_2:%.*]] = extractelement <3 x float> %p0, i64 2
  ; CHECK: [[UINT_2:%.*]] = call i32 @dx.op.legacyF32ToF16(i32 130, float [[FLOAT3_2]])
  ; CHECK: [[UINT3_0:%.*]] = insertelement <3 x i32> poison, i32 [[UINT_0]], i64 0
  ; CHECK: [[UINT3_1:%.*]] = insertelement <3 x i32> [[UINT3_0]], i32 [[UINT_1]], i64 1
  ; CHECK: [[UINT3_2:%.*]] = insertelement <3 x i32> [[UINT3_1]], i32 [[UINT_2]], i64 2
  ; CHECK : ret <3 x i32>  [[UINT3_2]]
  %hlsl.f32tof16 = tail call <3 x i32> @llvm.dx.legacyf32tof16.v3f32(<3 x float> %p0)
  ret <3 x i32> %hlsl.f32tof16
}

define hidden noundef <4 x i32> @_Z10test_uint4Dv4_j(<4 x float> noundef %p0) local_unnamed_addr #0 {
entry:
  ; CHECK: [[FLOAT4_0:%.*]] = extractelement <4 x float> %p0, i64 0
  ; CHECK: [[UINT_0:%.*]] = call i32 @dx.op.legacyF32ToF16(i32 130, float [[FLOAT4_0]])
  ; CHECK: [[FLOAT4_1:%.*]] = extractelement <4 x float> %p0, i64 1
  ; CHECK: [[UINT_1:%.*]] = call i32 @dx.op.legacyF32ToF16(i32 130, float [[FLOAT4_1]])
  ; CHECK: [[FLOAT4_2:%.*]] = extractelement <4 x float> %p0, i64 2
  ; CHECK: [[UINT_2:%.*]] = call i32 @dx.op.legacyF32ToF16(i32 130, float [[FLOAT4_2]])
  ; CHECK: [[FLOAT4_3:%.*]] = extractelement <4 x float> %p0, i64 3
  ; CHECK: [[UINT_3:%.*]] = call i32 @dx.op.legacyF32ToF16(i32 130, float [[FLOAT4_3]])
  ; CHECK: [[UINT4_0:%.*]] = insertelement <4 x i32> poison, i32 [[UINT_0]], i64 0
  ; CHECK: [[UINT4_1:%.*]] = insertelement <4 x i32> [[UINT4_0]], i32 [[UINT_1]], i64 1
  ; CHECK: [[UINT4_2:%.*]] = insertelement <4 x i32> [[UINT4_1]], i32 [[UINT_2]], i64 2
  ; CHECK: [[UINT4_3:%.*]] = insertelement <4 x i32> [[UINT4_2]], i32 [[UINT_3]], i64 3
  ; CHECK : ret <4 x i32>  [[UINT4_3]]
  %hlsl.f32tof16 = tail call <4 x i32> @llvm.dx.legacyf32tof16.v4i32(<4 x float> %p0)
  ret <4 x i32> %hlsl.f32tof16
}
