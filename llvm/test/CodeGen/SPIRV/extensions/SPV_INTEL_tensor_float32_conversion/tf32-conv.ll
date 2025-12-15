; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_tensor_float32_conversion %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_tensor_float32_conversion %s -o - -filetype=obj | spirv-val %}

; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: the builtin requires the following SPIR-V extension: SPV_INTEL_tensor_float32_conversion 

; CHECK: OpCapability TensorFloat32RoundingINTEL
; CHECK: OpExtension "SPV_INTEL_tensor_float32_conversion"

; CHECK-DAG: %[[VoidTy:.*]] = OpTypeVoid
; CHECK-DAG: %[[FP32Ty:.*]] = OpTypeFloat 32
; CHECK-DAG: %[[VecFloat2:.*]] = OpTypeVector %[[FP32Ty]] 2
; CHECK-DAG: %[[VecFloat3:.*]] = OpTypeVector %[[FP32Ty]] 3
; CHECK-DAG: %[[VecFloat4:.*]] = OpTypeVector %[[FP32Ty]] 4
; CHECK-DAG: %[[VecFloat8:.*]] = OpTypeVector %[[FP32Ty]] 8
; CHECK-DAG: %[[VecFloat16:.*]] = OpTypeVector %[[FP32Ty]] 16
; CHECK-DAG: %[[FloatConstId:.*]] = OpConstant %[[FP32Ty]] 1.5

; CHECK: OpFunction %[[VoidTy]]
; CHECK: %[[FP32ValId:.*]] = OpFunctionParameter %[[FP32Ty]]
; CHECK: %[[FP32v8ValId:.*]] = OpFunctionParameter %[[VecFloat8]]
; CHECK: OpRoundFToTF32INTEL %[[FP32Ty]] %[[FP32ValId]]
; CHECK: OpRoundFToTF32INTEL %[[VecFloat8]] %[[FP32v8ValId]]
; CHECK: OpRoundFToTF32INTEL %[[FP32Ty]] %[[FloatConstId]]

; CHECK: OpRoundFToTF32INTEL %[[FP32Ty]] 
; CHECK: OpRoundFToTF32INTEL %[[VecFloat2]]
; CHECK: OpRoundFToTF32INTEL %[[VecFloat3]]
; CHECK: OpRoundFToTF32INTEL %[[VecFloat4]]
; CHECK: OpRoundFToTF32INTEL %[[VecFloat8]]
; CHECK: OpRoundFToTF32INTEL %[[VecFloat16]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @test(float %a, <8 x float> %in) {
  %res1 = tail call spir_func float @_Z25__spirv_RoundFToTF32INTELf(float %a)
  %res2 = tail call spir_func <8 x float> @_Z25__spirv_RoundFToTF32INTELDv8_f(<8 x float> %in)
  %res3 = tail call spir_func float @_Z25__spirv_RoundFToTF32INTELf(float 1.500000e+00)
  ret void
}

declare spir_func float @_Z25__spirv_RoundFToTF32INTELf(float)
declare spir_func <8 x float> @_Z25__spirv_RoundFToTF32INTELDv8_f(<8 x float>)

define dso_local spir_kernel void @test_ocl(float %a) {
entry:
  %res4 = call spir_func float @_Z35intel_round_as_tensor_float32_floatt(float 0.000000e+00)
  %res5 = call spir_func <2 x float> @_Z37intel_round_as_tensor_float322_float2Dv2_t(<2 x float> zeroinitializer)
  %res6 = call spir_func <3 x float> @_Z37intel_round_as_tensor_float323_float3Dv3_t(<3 x float> zeroinitializer)
  %res7 = call spir_func <4 x float> @_Z37intel_round_as_tensor_float324_float4Dv4_t(<4 x float> zeroinitializer)
  %res8 = call spir_func <8 x float> @_Z37intel_round_as_tensor_float328_float8Dv8_t(<8 x float> zeroinitializer)
  %res9 = call spir_func <16 x float> @_Z39intel_round_as_tensor_float3216_float16Dv16_t(<16 x float> zeroinitializer)
  ret void
}

declare spir_func float @_Z35intel_round_as_tensor_float32_floatt(float)
declare spir_func <2 x float> @_Z37intel_round_as_tensor_float322_float2Dv2_t(<2 x float>)
declare spir_func <3 x float> @_Z37intel_round_as_tensor_float323_float3Dv3_t(<3 x float>)
declare spir_func <4 x float> @_Z37intel_round_as_tensor_float324_float4Dv4_t(<4 x float>)
declare spir_func <8 x float> @_Z37intel_round_as_tensor_float328_float8Dv8_t(<8 x float>)
declare spir_func <16 x float> @_Z39intel_round_as_tensor_float3216_float16Dv16_t(<16 x float>)
