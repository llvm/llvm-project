; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[SET:%.*]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: [[UINT:%.*]] = OpTypeInt 32 0
; CHECK-DAG: [[UINT2:%.*]] = OpTypeVector [[UINT]] 2
; CHECK-DAG: [[UINT3:%.*]] = OpTypeVector [[UINT]] 3
; CHECK-DAG: [[UINT4:%.*]] = OpTypeVector [[UINT]] 4
; CHECK-DAG: [[FLOAT:%.*]] = OpTypeFloat 32
; CHECK-DAG: [[FLOAT2:%.*]] = OpTypeVector [[FLOAT]] 2
; CHECK-DAG: [[FLOAT3:%.*]] = OpTypeVector [[FLOAT]] 3
; CHECK-DAG: [[FLOAT4:%.*]] = OpTypeVector [[FLOAT]] 4

; CHECK: [[P0:%.*]] = OpFunctionParameter [[UINT]]
; CHECK: [[UNPACK2:%.*]] = OpExtInst [[FLOAT2]] [[SET]] UnpackHalf2x16 [[P0]]
; CHECK: [[UNPACK:%.*]] = OpCompositeExtract [[FLOAT]] [[UNPACK2]] 0
; CHECK: OpReturnValue [[UNPACK]]
define hidden noundef nofpclass(nan inf) float @_Z11test_scalarj(i32 noundef %p0) local_unnamed_addr #0 {
entry:
  %hlsl.f16tof32 = tail call reassoc nnan ninf nsz arcp afn float @llvm.spv.legacyf16tof32.i32(i32 %p0)
  ret float %hlsl.f16tof32
}

; CHECK: [[P0:%.*]] = OpFunctionParameter [[UINT2]]
; CHECK-DAG: [[P0_0:%.*]] = OpCompositeExtract [[UINT]] [[P0]] 0
; CHECK-DAG: [[P0_1:%.*]] = OpCompositeExtract [[UINT]] [[P0]] 1
; CHECK-DAG: [[UNPACK2_0:%.*]] = OpExtInst [[FLOAT2]] [[SET]] UnpackHalf2x16 [[P0_0]]
; CHECK-DAG: [[UNPACK2_1:%.*]] = OpExtInst [[FLOAT2]] [[SET]] UnpackHalf2x16 [[P0_1]]
; CHECK-DAG: [[RESULT_0:%.*]] = OpCompositeExtract [[FLOAT]] [[UNPACK2_0]] 0
; CHECK-DAG: [[RESULT_1:%.*]] = OpCompositeExtract [[FLOAT]] [[UNPACK2_1]] 0
; CHECK: [[RESULT:%.*]] = OpCompositeConstruct [[FLOAT2]] [[RESULT_0]] [[RESULT_1]]
; CHECK: OpReturnValue [[RESULT]]
define hidden noundef nofpclass(nan inf) <2 x float> @_Z10test_uint2Dv2_j(<2 x i32> noundef %p0) local_unnamed_addr #0 {
entry:
  %hlsl.f16tof32 = tail call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.legacyf16tof32.v2i32(<2 x i32> %p0)
  ret <2 x float> %hlsl.f16tof32
}

; CHECK: [[P0:%.*]] = OpFunctionParameter [[UINT3]]
; CHECK-DAG: [[P0_0:%.*]] = OpCompositeExtract [[UINT]] [[P0]] 0
; CHECK-DAG: [[P0_1:%.*]] = OpCompositeExtract [[UINT]] [[P0]] 1
; CHECK-DAG: [[P0_2:%.*]] = OpCompositeExtract [[UINT]] [[P0]] 2
; CHECK-DAG: [[UNPACK3_0:%.*]] = OpExtInst [[FLOAT2]] [[SET]] UnpackHalf2x16 [[P0_0]]
; CHECK-DAG: [[UNPACK3_1:%.*]] = OpExtInst [[FLOAT2]] [[SET]] UnpackHalf2x16 [[P0_1]]
; CHECK-DAG: [[UNPACK3_2:%.*]] = OpExtInst [[FLOAT2]] [[SET]] UnpackHalf2x16 [[P0_2]]
; CHECK-DAG: [[RESULT_0:%.*]] = OpCompositeExtract [[FLOAT]] [[UNPACK3_0]] 0
; CHECK-DAG: [[RESULT_1:%.*]] = OpCompositeExtract [[FLOAT]] [[UNPACK3_1]] 0
; CHECK-DAG: [[RESULT_2:%.*]] = OpCompositeExtract [[FLOAT]] [[UNPACK3_2]] 0
; CHECK: [[RESULT:%.*]] = OpCompositeConstruct [[FLOAT3]] [[RESULT_0]] [[RESULT_1]] [[RESULT_2]]
; CHECK: OpReturnValue [[RESULT]]
define hidden noundef nofpclass(nan inf) <3 x float> @_Z10test_uint3Dv3_j(<3 x i32> noundef %p0) local_unnamed_addr #0 {
entry:
  %hlsl.f16tof32 = tail call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.spv.legacyf16tof32.v3i32(<3 x i32> %p0)
  ret <3 x float> %hlsl.f16tof32
}

; CHECK: [[P0:%.*]] = OpFunctionParameter [[UINT4]]
; CHECK-DAG: [[P0_0:%.*]] = OpCompositeExtract [[UINT]] [[P0]] 0
; CHECK-DAG: [[P0_1:%.*]] = OpCompositeExtract [[UINT]] [[P0]] 1
; CHECK-DAG: [[P0_2:%.*]] = OpCompositeExtract [[UINT]] [[P0]] 2
; CHECK-DAG: [[P0_3:%.*]] = OpCompositeExtract [[UINT]] [[P0]] 3
; CHECK-DAG: [[UNPACK4_0:%.*]] = OpExtInst [[FLOAT2]] [[SET]] UnpackHalf2x16 [[P0_0]]
; CHECK-DAG: [[UNPACK4_1:%.*]] = OpExtInst [[FLOAT2]] [[SET]] UnpackHalf2x16 [[P0_1]]
; CHECK-DAG: [[UNPACK4_2:%.*]] = OpExtInst [[FLOAT2]] [[SET]] UnpackHalf2x16 [[P0_2]]
; CHECK-DAG: [[UNPACK4_3:%.*]] = OpExtInst [[FLOAT2]] [[SET]] UnpackHalf2x16 [[P0_3]]
; CHECK-DAG: [[RESULT_0:%.*]] = OpCompositeExtract [[FLOAT]] [[UNPACK4_0]] 0
; CHECK-DAG: [[RESULT_1:%.*]] = OpCompositeExtract [[FLOAT]] [[UNPACK4_1]] 0
; CHECK-DAG: [[RESULT_2:%.*]] = OpCompositeExtract [[FLOAT]] [[UNPACK4_2]] 0
; CHECK-DAG: [[RESULT_3:%.*]] = OpCompositeExtract [[FLOAT]] [[UNPACK4_3]] 0
; CHECK: [[RESULT:%.*]] = OpCompositeConstruct [[FLOAT4]] [[RESULT_0]] [[RESULT_1]] [[RESULT_2]] [[RESULT_3]]
; CHECK: OpReturnValue [[RESULT]]
define hidden noundef nofpclass(nan inf) <4 x float> @_Z10test_uint4Dv4_j(<4 x i32> noundef %p0) local_unnamed_addr #0 {
entry:
  %hlsl.f16tof32 = tail call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.legacyf16tof32.v4i32(<4 x i32> %p0)
  ret <4 x float> %hlsl.f16tof32
}
