; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName [[TRUNC32_16v1:%.*]] "i32toi16v1"
; CHECK-DAG: OpName [[TRUNC32_16v17:%.*]] "i32toi16v17"
; CHECK-DAG: OpName [[SEXT8_32v1:%.*]] "s8tos32v1"
; CHECK-DAG: OpName [[SEXT8_32v17:%.*]] "s8tos32v17"
; CHECK-DAG: OpName [[ZEXT8_32v1:%.*]] "u8tou32v1"
; CHECK-DAG: OpName [[ZEXT8_32v17:%.*]] "u8tou32v17"

; CHECK-DAG: [[U32:%.*]] = OpTypeInt 32 0
; CHECK-DAG: [[U16:%.*]] = OpTypeInt 16 0
; CHECK-DAG: [[U8:%.*]] = OpTypeInt 8 0
; CHECK-DAG: [[ONE:%.*]] = OpConstant [[U32]] 1
; CHECK-DAG: [[SEVENTEEN:%.*]] = OpConstant [[U32]] 17
; CHECK-DAG: [[U32v1:%.*]] = OpTypeVectorIdEXT [[U32]] [[ONE]]
; CHECK-DAG: [[U32v17:%.*]] = OpTypeVectorIdEXT [[U32]] [[SEVENTEEN]]
; CHECK-DAG: [[U16v1:%.*]] = OpTypeVectorIdEXT [[U16]] [[ONE]]
; CHECK-DAG: [[U16v17:%.*]] = OpTypeVectorIdEXT [[U16]] [[SEVENTEEN]]
; CHECK-DAG: [[U8v1:%.*]] = OpTypeVectorIdEXT [[U8]] [[ONE]]
; CHECK-DAG: [[U8v17:%.*]] = OpTypeVectorIdEXT [[U8]] [[SEVENTEEN]]

; CHECK:      [[TRUNC32_16v1]] = OpFunction [[U16v1]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32v1]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U16v1]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <1 x i16> @i32toi16v1(<1 x i32> %a) {
    %r = trunc <1 x i32> %a to <1 x i16>
    ret <1 x i16> %r
}

; CHECK:      [[TRUNC32_16v17]] = OpFunction [[U16v17]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32v17]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U16v17]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <17 x i16> @i32toi16v17(<17 x i32> %a) {
    %r = trunc <17 x i32> %a to <17 x i16>
    ret <17 x i16> %r
}

; CHECK:      [[SEXT8_32v1]] = OpFunction [[U32v1]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v1]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpSConvert [[U32v1]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <1 x i32>  @s8tos32v1(<1 x i8> %a) {
  %r = sext <1 x i8> %a to <1 x i32>
  ret <1 x i32>  %r
}

; CHECK:      [[SEXT8_32v17]] = OpFunction [[U32v17]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v17]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpSConvert [[U32v17]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <17 x i32>  @s8tos32v17(<17 x i8> %a) {
  %r = sext <17 x i8> %a to <17 x i32>
  ret <17 x i32>  %r
}

; CHECK:      [[ZEXT8_32v1]] = OpFunction [[U32v1]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v1]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U32v1]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <1 x i32>  @u8tou32v1(<1 x i8> %a) {
  %r = zext <1 x i8> %a to <1 x i32>
  ret <1 x i32>  %r
}

; CHECK:      [[ZEXT8_32v17]] = OpFunction [[U32v17]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v17]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpUConvert [[U32v17]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <17 x i32>  @u8tou32v17(<17 x i8> %a) {
  %r = zext <17 x i8> %a to <17 x i32>
  ret <17 x i32>  %r
}
