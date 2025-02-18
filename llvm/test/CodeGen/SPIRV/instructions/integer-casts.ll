; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName [[TRUNC32_16:%.*]] "i32toi16"
; CHECK-DAG: OpName [[TRUNC32_8:%.*]] "i32toi8"
; CHECK-DAG: OpName [[TRUNC16_8:%.*]] "i16toi8"
; CHECK-DAG: OpName [[SEXT8_32:%.*]] "s8tos32"
; CHECK-DAG: OpName [[SEXT8_16:%.*]] "s8tos16"
; CHECK-DAG: OpName [[SEXT16_32:%.*]] "s16tos32"
; CHECK-DAG: OpName [[ZEXT8_32:%.*]] "u8tou32"
; CHECK-DAG: OpName [[ZEXT8_16:%.*]] "u8tou16"
; CHECK-DAG: OpName [[ZEXT16_32:%.*]] "u16tou32"

; CHECK-DAG: OpName %[[#R17:]] "r17"
; CHECK-DAG: OpName %[[#R18:]] "r18"
; CHECK-DAG: OpName %[[#R19:]] "r19"
; CHECK-DAG: OpName %[[#R20:]] "r20"
; CHECK-DAG: OpName %[[#R21:]] "r21"

; CHECK-DAG: OpName [[TRUNC32_16v4:%.*]] "i32toi16v4"
; CHECK-DAG: OpName [[TRUNC32_8v4:%.*]] "i32toi8v4"
; CHECK-DAG: OpName [[TRUNC16_8v4:%.*]] "i16toi8v4"
; CHECK-DAG: OpName [[SEXT8_32v4:%.*]] "s8tos32v4"
; CHECK-DAG: OpName [[SEXT8_16v4:%.*]] "s8tos16v4"
; CHECK-DAG: OpName [[SEXT16_32v4:%.*]] "s16tos32v4"
; CHECK-DAG: OpName [[ZEXT8_32v4:%.*]] "u8tou32v4"
; CHECK-DAG: OpName [[ZEXT8_16v4:%.*]] "u8tou16v4"
; CHECK-DAG: OpName [[ZEXT16_32v4:%.*]] "u16tou32v4"

; CHECK-DAG: OpDecorate %[[#R17]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#R18]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#R19]] FPRoundingMode RTP
; CHECK-DAG: OpDecorate %[[#R20]] FPRoundingMode RTN
; CHECK-DAG: OpDecorate %[[#R21]] SaturatedConversion

; CHECK-DAG: [[F32:%.*]] = OpTypeFloat 32
; CHECK-DAG: [[F16:%.*]] = OpTypeFloat 16
; CHECK-DAG: [[U64:%.*]] = OpTypeInt 64 0
; CHECK-DAG: [[U32:%.*]] = OpTypeInt 32 0
; CHECK-DAG: [[U16:%.*]] = OpTypeInt 16 0
; CHECK-DAG: [[U8:%.*]] = OpTypeInt 8 0
; CHECK-DAG: [[F32v2:%.*]] = OpTypeVector [[F32]] 2
; CHECK-DAG: [[U32v4:%.*]] = OpTypeVector [[U32]] 4
; CHECK-DAG: [[U16v4:%.*]] = OpTypeVector [[U16]] 4
; CHECK-DAG: [[U8v4:%.*]] = OpTypeVector [[U8]] 4


; CHECK:      [[TRUNC32_16]] = OpFunction [[U16]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U16]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i16 @i32toi16(i32 %a) {
    %r = trunc i32 %a to i16
    ret i16 %r
}

; CHECK:      [[TRUNC32_8]] = OpFunction [[U8]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U8]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i8 @i32toi8(i32 %a) {
    %r = trunc i32 %a to i8
    ret i8 %r
}

; CHECK:      [[TRUNC16_8]] = OpFunction [[U8]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U8]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i8 @i16toi8(i16 %a) {
    %r = trunc i16 %a to i8
    ret i8 %r
}


; CHECK:      [[SEXT8_32]] = OpFunction [[U32]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpSConvert [[U32]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @s8tos32(i8 %a) {
  %r = sext i8 %a to i32
  ret i32 %r
}

; CHECK:      [[SEXT8_16]] = OpFunction [[U16]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpSConvert [[U16]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i16 @s8tos16(i8 %a) {
  %r = sext i8 %a to i16
  ret i16 %r
}

; CHECK:      [[SEXT16_32]] = OpFunction [[U32]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpSConvert [[U32]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @s16tos32(i16 %a) {
  %r = sext i16 %a to i32
  ret i32 %r
}

; CHECK:      [[ZEXT8_32]] = OpFunction [[U32]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U32]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @u8tou32(i8 %a) {
  %r = zext i8 %a to i32
  ret i32 %r
}

; CHECK:      [[ZEXT8_16]] = OpFunction [[U16]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U16]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i16 @u8tou16(i8 %a) {
  %r = zext i8 %a to i16
  ret i16 %r
}

; CHECK:      [[ZEXT16_32]] = OpFunction [[U32]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U32]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @u16tou32(i16 %a) {
  %r = zext i16 %a to i32
  ret i32 %r
}

; CHECK:      [[TRUNC32_16v4]] = OpFunction [[U16v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32v4]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U16v4]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i16> @i32toi16v4(<4 x i32> %a) {
    %r = trunc <4 x i32> %a to <4 x i16>
    ret <4 x i16> %r
}

; CHECK:      [[TRUNC32_8v4]] = OpFunction [[U8v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32v4]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U8v4]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i8> @i32toi8v4(<4 x i32> %a) {
    %r = trunc <4 x i32> %a to <4 x i8>
    ret <4 x i8> %r
}

; CHECK:      [[TRUNC16_8v4]] = OpFunction [[U8v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16v4]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U8v4]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i8> @i16toi8v4(<4 x i16> %a) {
    %r = trunc <4 x i16> %a to <4 x i8>
    ret <4 x i8> %r
}


; CHECK:      [[SEXT8_32v4]] = OpFunction [[U32v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v4]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpSConvert [[U32v4]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i32>  @s8tos32v4(<4 x i8> %a) {
  %r = sext <4 x i8> %a to <4 x i32>
  ret <4 x i32>  %r
}

; CHECK:      [[SEXT8_16v4]] = OpFunction [[U16v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v4]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpSConvert [[U16v4]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i16> @s8tos16v4(<4 x i8> %a) {
  %r = sext <4 x i8> %a to <4 x i16>
  ret <4 x i16> %r
}

; CHECK:      [[SEXT16_32v4]] = OpFunction [[U32v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16v4]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpSConvert [[U32v4]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i32>  @s16tos32v4(<4 x i16> %a) {
  %r = sext <4 x i16> %a to <4 x i32>
  ret <4 x i32>  %r
}

; CHECK:      [[ZEXT8_32v4]] = OpFunction [[U32v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v4]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U32v4]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i32>  @u8tou32v4(<4 x i8> %a) {
  %r = zext <4 x i8> %a to <4 x i32>
  ret <4 x i32>  %r
}

; CHECK:      [[ZEXT8_16v4]] = OpFunction [[U16v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v4]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U16v4]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i16> @u8tou16v4(<4 x i8> %a) {
  %r = zext <4 x i8> %a to <4 x i16>
  ret <4 x i16> %r
}

; CHECK:      [[ZEXT16_32v4]] = OpFunction [[U32v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16v4]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U32v4]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i32>  @u16tou32v4(<4 x i16> %a) {
  %r = zext <4 x i16> %a to <4 x i32>
  ret <4 x i32>  %r
}

; CHECK: OpFunction
; CHECK: [[Arg1:%.*]] = OpFunctionParameter
; CHECK: [[Arg2:%.*]] = OpFunctionParameter
; CHECK: %[[#]] = OpConvertFToU [[U32]] %[[#]]
; CHECK: %[[#]] = OpConvertFToS [[U32]] %[[#]]
; CHECK: %[[#]] = OpConvertSToF [[F32]] %[[#]]
; CHECK: %[[#]] = OpConvertUToF [[F32]] %[[#]]
; CHECK: %[[#]] = OpUConvert [[U32]] %[[#]]
; CHECK: %[[#]] = OpSConvert [[U32]] %[[#]]
; CHECK: %[[#]] = OpFConvert [[F16]] %[[#]]
; CHECK: %[[#]] = OpQuantizeToF16 [[F32]] %[[#]]
; CHECK: %[[#]] = OpSatConvertSToU [[U64]] %[[#]]
; CHECK: %[[#]] = OpSatConvertUToS [[U64]] %[[#]]
; CHECK: %[[#]] = OpConvertPtrToU [[U64]] [[Arg1]]
; CHECK: %[[#]] = OpConvertUToPtr %[[#]] [[Arg2]]
; CHECK: %[[#]] = OpUConvert [[U32v4]] %[[#]]
; CHECK: %[[#]] = OpSConvert [[U32v4]] %[[#]]
; CHECK: %[[#]] = OpConvertUToF [[F32]] %[[#]]
; CHECK: %[[#]] = OpConvertUToF [[F32]] %[[#]]
; CHECK: %[[#R17]] = OpFConvert [[F32v2]] %[[#]]
; CHECK: %[[#R18]] = OpFConvert [[F32v2]] %[[#]]
; CHECK: %[[#R19]] = OpFConvert [[F32v2]] %[[#]]
; CHECK: %[[#R20]] = OpFConvert [[F32v2]] %[[#]]
; CHECK: %[[#R21]] = OpConvertFToU [[U8]] %[[#]]
; CHECK: OpFunctionEnd
define dso_local spir_kernel void @test_wrappers(ptr addrspace(4) %arg, i64 %arg_ptr, <4 x i8> %arg_v2) {
  %r1 = call spir_func i32 @__spirv_ConvertFToU(float 0.000000e+00)
  %r2 = call spir_func i32 @__spirv_ConvertFToS(float 0.000000e+00)
  %r3 = call spir_func float @__spirv_ConvertSToF(i32 1)
  %r4 = call spir_func float @__spirv_ConvertUToF(i32 1)
  %r5 = call spir_func i32 @__spirv_UConvert(i64 1)
  %r6 = call spir_func i32 @__spirv_SConvert(i64 1)
  %r7 = call spir_func half @__spirv_FConvert(float 0.000000e+00)
  %r8 = call spir_func float @__spirv_QuantizeToF16(float 0.000000e+00)
  %r9 = call spir_func i64 @__spirv_SatConvertSToU(i64 1)
  %r10 = call spir_func i64 @__spirv_SatConvertUToS(i64 1)
  %r11 = call spir_func i64 @__spirv_ConvertPtrToU(ptr addrspace(4) %arg)
  %r12 = call spir_func ptr addrspace(4) @__spirv_ConvertUToPtr(i64 %arg_ptr)
  %r13 = call spir_func <4 x i32> @_Z22__spirv_UConvert_Rint2Dv2_a(<4 x i8> %arg_v2)
  %r14 = call spir_func <4 x i32> @_Z22__spirv_SConvert_Rint2Dv2_a(<4 x i8> %arg_v2)
  %r15 = call spir_func float @_Z30__spirv_ConvertUToF_Rfloat_rtz(i64 %arg_ptr)
  %r16 = call spir_func float @__spirv_ConvertUToF_Rfloat_rtz(i64 %arg_ptr)
  %r17 = call spir_func <2 x float> @_Z28__spirv_FConvert_Rfloat2_rtzDv2_DF16_(<2 x half> noundef <half 0xH409A, half 0xH439A>)
  %r18 = call spir_func <2 x float> @_Z28__spirv_FConvert_Rfloat2_rteDv2_DF16_(<2 x half> noundef <half 0xH409A, half 0xH439A>)
  %r19 = call spir_func <2 x float> @_Z28__spirv_FConvert_Rfloat2_rtpDv2_DF16_(<2 x half> noundef <half 0xH409A, half 0xH439A>)
  %r20 = call spir_func <2 x float> @_Z28__spirv_FConvert_Rfloat2_rtnDv2_DF16_(<2 x half> noundef <half 0xH409A, half 0xH439A>)
  %r21 = call spir_func i8 @_Z30__spirv_ConvertFToU_Ruchar_satf(float noundef 42.0)
  ret void
}

declare dso_local spir_func i32 @__spirv_ConvertFToU(float)
declare dso_local spir_func i32 @__spirv_ConvertFToS(float)
declare dso_local spir_func float @__spirv_ConvertSToF(i32)
declare dso_local spir_func float @__spirv_ConvertUToF(i32)
declare dso_local spir_func i32 @__spirv_UConvert(i64)
declare dso_local spir_func i32 @__spirv_SConvert(i64)
declare dso_local spir_func half @__spirv_FConvert(float)
declare dso_local spir_func float @__spirv_QuantizeToF16(float)
declare dso_local spir_func i64 @__spirv_SatConvertSToU(i64)
declare dso_local spir_func i64 @__spirv_SatConvertUToS(i64)
declare dso_local spir_func i64 @__spirv_ConvertPtrToU(ptr addrspace(4))
declare dso_local spir_func ptr addrspace(4) @__spirv_ConvertUToPtr(i64)
declare dso_local spir_func <4 x i32> @_Z22__spirv_UConvert_Rint2Dv2_a(<4 x i8>)
declare dso_local spir_func <4 x i32> @_Z22__spirv_SConvert_Rint2Dv2_a(<4 x i8>)
declare dso_local spir_func float @_Z30__spirv_ConvertUToF_Rfloat_rtz(i64)
declare dso_local spir_func float @__spirv_ConvertUToF_Rfloat_rtz(i64)
declare dso_local spir_func <2 x float> @_Z28__spirv_FConvert_Rfloat2_rtzDv2_DF16_(<2 x half> noundef)
declare dso_local spir_func <2 x float> @_Z28__spirv_FConvert_Rfloat2_rteDv2_DF16_(<2 x half> noundef)
declare dso_local spir_func <2 x float> @_Z28__spirv_FConvert_Rfloat2_rtpDv2_DF16_(<2 x half> noundef)
declare dso_local spir_func <2 x float> @_Z28__spirv_FConvert_Rfloat2_rtnDv2_DF16_(<2 x half> noundef)
declare dso_local spir_func i8 @_Z30__spirv_ConvertFToU_Ruchar_satf(float)
