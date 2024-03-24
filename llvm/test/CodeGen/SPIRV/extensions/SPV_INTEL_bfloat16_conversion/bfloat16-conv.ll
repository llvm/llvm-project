; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-extensions=SPV_INTEL_bfloat16_conversion %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-extensions=SPV_INTEL_bfloat16_conversion %s -o - -filetype=obj | spirv-val %}

; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: the builtin requires the following SPIR-V extension: SPV_INTEL_bfloat16_conversion

; CHECK: OpCapability BFloat16ConversionINTEL
; CHECK: OpExtension "SPV_INTEL_bfloat16_conversion"

; CHECK-DAG: %[[VoidTy:.*]] = OpTypeVoid
; CHECK-DAG: %[[Int16Ty:.*]] = OpTypeInt 16 0
; CHECK-DAG: %[[FP32Ty:.*]] = OpTypeFloat 32
; CHECK-DAG: %[[VecFloat2:.*]] = OpTypeVector %[[FP32Ty]] 2
; CHECK-DAG: %[[VecInt162:.*]] = OpTypeVector %[[Int16Ty]] 2
; CHECK-DAG: %[[VecFloat3:.*]] = OpTypeVector %[[FP32Ty]] 3
; CHECK-DAG: %[[VecInt163:.*]] = OpTypeVector %[[Int16Ty]] 3
; CHECK-DAG: %[[VecFloat4:.*]] = OpTypeVector %[[FP32Ty]] 4
; CHECK-DAG: %[[VecInt164:.*]] = OpTypeVector %[[Int16Ty]] 4
; CHECK-DAG: %[[VecFloat8:.*]] = OpTypeVector %[[FP32Ty]] 8
; CHECK-DAG: %[[VecInt168:.*]] = OpTypeVector %[[Int16Ty]] 8
; CHECK-DAG: %[[VecFloat16:.*]] = OpTypeVector %[[FP32Ty]] 16
; CHECK-DAG: %[[VecInt1616:.*]] = OpTypeVector %[[Int16Ty]] 16
; CHECK-DAG: %[[IntConstId:.*]] = OpConstant %[[Int16Ty]] 67
; CHECK-DAG: %[[FloatConstId:.*]] = OpConstant %[[FP32Ty]] 1.5

; CHECK: OpFunction %[[VoidTy]]
; CHECK: %[[FP32ValId:.*]] = OpFunctionParameter %[[FP32Ty]]
; CHECK: %[[FP32v8ValId:.*]] = OpFunctionParameter %[[VecFloat8]]

; CHECK: %[[Int16ValId:.*]] = OpConvertFToBF16INTEL %[[Int16Ty]] %[[FP32ValId]]
; CHECK: OpConvertBF16ToFINTEL %[[FP32Ty]] %[[Int16ValId]]
; CHECK: %[[Int16v8ValId:.*]] = OpConvertFToBF16INTEL %[[VecInt168]] %[[FP32v8ValId]]
; CHECK: OpConvertBF16ToFINTEL %[[VecFloat8]] %[[Int16v8ValId]]
; CHECK: OpConvertFToBF16INTEL %[[Int16Ty]] %[[FloatConstId]]
; CHECK: OpConvertBF16ToFINTEL %[[FP32Ty]] %[[IntConstId]]

; CHECK: OpConvertFToBF16INTEL %[[Int16Ty]]
; CHECK: OpConvertFToBF16INTEL %[[VecInt162]]
; CHECK: OpConvertFToBF16INTEL %[[VecInt163]]
; CHECK: OpConvertFToBF16INTEL %[[VecInt164]]
; CHECK: OpConvertFToBF16INTEL %[[VecInt168]]
; CHECK: OpConvertFToBF16INTEL %[[VecInt1616]]
; CHECK: OpConvertBF16ToFINTEL %[[FP32Ty]]
; CHECK: OpConvertBF16ToFINTEL %[[VecFloat2]]
; CHECK: OpConvertBF16ToFINTEL %[[VecFloat3]]
; CHECK: OpConvertBF16ToFINTEL %[[VecFloat4]]
; CHECK: OpConvertBF16ToFINTEL %[[VecFloat8]]
; CHECK: OpConvertBF16ToFINTEL %[[VecFloat16]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @test(float %a, <8 x float> %in) {
  %res1 = tail call spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float %a)
  %res2 = tail call spir_func float @_Z27__spirv_ConvertBF16ToFINTELs(i16 zeroext %res1)
  %res3 = tail call spir_func <8 x i16> @_Z27__spirv_ConvertFToBF16INTELDv8_f(<8 x float> %in)
  %res4 = tail call spir_func <8 x float> @_Z27__spirv_ConvertBF16ToFINTELDv8_s(<8 x i16> %res3)
  %res5 = tail call spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float 1.500000e+00)
  %res6 = tail call spir_func float @_Z27__spirv_ConvertBF16ToFINTELs(i16 67)
  ret void
}

declare spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float)
declare spir_func float @_Z27__spirv_ConvertBF16ToFINTELs(i16 zeroext)
declare spir_func <8 x i16> @_Z27__spirv_ConvertFToBF16INTELDv8_f(<8 x float>)
declare spir_func <8 x float> @_Z27__spirv_ConvertBF16ToFINTELDv8_s(<8 x i16>)

define dso_local spir_kernel void @test_ocl() {
entry:
  %res = call spir_func zeroext i16 @_Z32intel_convert_bfloat16_as_ushortf(float 0.000000e+00)
  %res1 = call spir_func <2 x i16> @_Z34intel_convert_bfloat162_as_ushort2Dv2_f(<2 x float> zeroinitializer)
  %res2 = call spir_func <3 x i16> @_Z34intel_convert_bfloat163_as_ushort3Dv3_f(<3 x float> zeroinitializer)
  %res3 = call spir_func <4 x i16> @_Z34intel_convert_bfloat164_as_ushort4Dv4_f(<4 x float> zeroinitializer)
  %res4 = call spir_func <8 x i16> @_Z34intel_convert_bfloat168_as_ushort8Dv8_f(<8 x float> zeroinitializer)
  %res5 = call spir_func <16 x i16> @_Z36intel_convert_bfloat1616_as_ushort16Dv16_f(<16 x float> zeroinitializer)
  %res6 = call spir_func float @_Z31intel_convert_as_bfloat16_floatt(i16 zeroext 0)
  %res7 = call spir_func <2 x float> @_Z33intel_convert_as_bfloat162_float2Dv2_t(<2 x i16> zeroinitializer)
  %res8 = call spir_func <3 x float> @_Z33intel_convert_as_bfloat163_float3Dv3_t(<3 x i16> zeroinitializer)
  %res9 = call spir_func <4 x float> @_Z33intel_convert_as_bfloat164_float4Dv4_t(<4 x i16> zeroinitializer)
  %res10 = call spir_func <8 x float> @_Z33intel_convert_as_bfloat168_float8Dv8_t(<8 x i16> zeroinitializer)
  %res11 = call spir_func <16 x float> @_Z35intel_convert_as_bfloat1616_float16Dv16_t(<16 x i16> zeroinitializer)
  ret void
}

declare spir_func zeroext i16 @_Z32intel_convert_bfloat16_as_ushortf(float)
declare spir_func <2 x i16> @_Z34intel_convert_bfloat162_as_ushort2Dv2_f(<2 x float>)
declare spir_func <3 x i16> @_Z34intel_convert_bfloat163_as_ushort3Dv3_f(<3 x float>)
declare spir_func <4 x i16> @_Z34intel_convert_bfloat164_as_ushort4Dv4_f(<4 x float>)
declare spir_func <8 x i16> @_Z34intel_convert_bfloat168_as_ushort8Dv8_f(<8 x float>)
declare spir_func <16 x i16> @_Z36intel_convert_bfloat1616_as_ushort16Dv16_f(<16 x float>)
declare spir_func float @_Z31intel_convert_as_bfloat16_floatt(i16 zeroext)
declare spir_func <2 x float> @_Z33intel_convert_as_bfloat162_float2Dv2_t(<2 x i16>)
declare spir_func <3 x float> @_Z33intel_convert_as_bfloat163_float3Dv3_t(<3 x i16>)
declare spir_func <4 x float> @_Z33intel_convert_as_bfloat164_float4Dv4_t(<4 x i16>)
declare spir_func <8 x float> @_Z33intel_convert_as_bfloat168_float8Dv8_t(<8 x i16>)
declare spir_func <16 x float> @_Z35intel_convert_as_bfloat1616_float16Dv16_t(<16 x i16>)
