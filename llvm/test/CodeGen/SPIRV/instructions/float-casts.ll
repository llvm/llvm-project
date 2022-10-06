; RUN: llc -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName [[TRUNC32_16:%.*]] "f32tof16"
; CHECK-DAG: OpName [[EXT16_32:%.*]] "f16tof32"

; CHECK-DAG: OpName [[TRUNC32_16v3:%.*]] "f32tof16v3"
; CHECK-DAG: OpName [[EXT16_32v3:%.*]] "f16tof32v3"

; CHECK-DAG: OpName [[F32toS32:%.*]] "f32tos32"
; CHECK-DAG: OpName [[F32toS16:%.*]] "f32tos16"
; CHECK-DAG: OpName [[F32toS8:%.*]] "f32tos8"
; CHECK-DAG: OpName [[F16toS32:%.*]] "f16tos32"
; CHECK-DAG: OpName [[F16toS16:%.*]] "f16tos16"
; CHECK-DAG: OpName [[F16toS8:%.*]] "f16tos8"

; CHECK-DAG: OpName [[F32toU32v2:%.*]] "f32tou32v2"
; CHECK-DAG: OpName [[F32toU16v2:%.*]] "f32tou16v2"
; CHECK-DAG: OpName [[F32toU8v2:%.*]] "f32tou8v2"
; CHECK-DAG: OpName [[F16toU32v2:%.*]] "f16tou32v2"
; CHECK-DAG: OpName [[F16toU16v2:%.*]] "f16tou16v2"
; CHECK-DAG: OpName [[F16toU8v2:%.*]] "f16tou8v2"

; CHECK-DAG: [[F32:%.*]] = OpTypeFloat 32
; CHECK-DAG: [[F16:%.*]] = OpTypeFloat 16
; CHECK-DAG: [[F32v2:%.*]] = OpTypeVector [[F32]] 2
; CHECK-DAG: [[F16v2:%.*]] = OpTypeVector [[F16]] 2
; CHECK-DAG: [[F32v3:%.*]] = OpTypeVector [[F32]] 3
; CHECK-DAG: [[F16v3:%.*]] = OpTypeVector [[F16]] 3
; CHECK-DAG: [[U32:%.*]] = OpTypeInt 32 0
; CHECK-DAG: [[U16:%.*]] = OpTypeInt 16 0
; CHECK-DAG: [[U8:%.*]] = OpTypeInt 8 0
; CHECK-DAG: [[U32v2:%.*]] = OpTypeVector [[U32]] 2
; CHECK-DAG: [[U16v2:%.*]] = OpTypeVector [[U16]] 2
; CHECK-DAG: [[U8v2:%.*]] = OpTypeVector [[U8]] 2


; CHECK:      [[TRUNC32_16]] = OpFunction [[F16]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F32]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpFConvert [[F16]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define half @f32tof16(float %a) {
    %r = fptrunc float %a to half
    ret half %r
}

; CHECK:      [[EXT16_32]] = OpFunction [[F32]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F16]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpFConvert [[F32]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define float @f16tof32(half %a) {
  %r = fpext half %a to float
  ret float %r
}

; CHECK:      [[TRUNC32_16v3]] = OpFunction [[F16v3]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F32v3]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpFConvert [[F16v3]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x half> @f32tof16v3(<3 x float> %a) {
    %r = fptrunc <3 x float> %a to <3 x half>
    ret <3 x half> %r
}

; CHECK:      [[EXT16_32v3]] = OpFunction [[F32v3]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F16v3]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpFConvert [[F32v3]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x float> @f16tof32v3(<3 x half> %a) {
  %r = fpext <3 x half> %a to <3 x float>
  ret <3 x float> %r
}

; CHECK:      [[F32toS32]] = OpFunction [[U32]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F32]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToS [[U32]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @f32tos32(float %a) {
  %r = fptosi float %a to i32
  ret i32 %r
}

; CHECK:      [[F32toS16]] = OpFunction [[U16]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F32]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToS [[U16]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i16 @f32tos16(float %a) {
  %r = fptosi float %a to i16
  ret i16 %r
}

; CHECK:      [[F32toS8]] = OpFunction [[U8]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F32]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToS [[U8]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i8 @f32tos8(float %a) {
  %r = fptosi float %a to i8
  ret i8 %r
}

; CHECK:      [[F16toS32]] = OpFunction [[U32]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F16]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToS [[U32]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @f16tos32(half %a) {
  %r = fptosi half %a to i32
  ret i32 %r
}

; CHECK:      [[F16toS16]] = OpFunction [[U16]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F16]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToS [[U16]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i16 @f16tos16(half %a) {
  %r = fptosi half %a to i16
  ret i16 %r
}

; CHECK:      [[F16toS8]] = OpFunction [[U8]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F16]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToS [[U8]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i8 @f16tos8(half %a) {
  %r = fptosi half %a to i8
  ret i8 %r
}

; CHECK:      [[F32toU32v2]] = OpFunction [[U32v2]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F32v2]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToU [[U32v2]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x i32> @f32tou32v2(<2 x float> %a) {
  %r = fptoui <2 x float> %a to <2 x i32>
  ret <2 x i32> %r
}

; CHECK:      [[F32toU16v2]] = OpFunction [[U16v2]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F32v2]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToU [[U16v2]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x i16> @f32tou16v2(<2 x float> %a) {
  %r = fptoui <2 x float> %a to <2 x i16>
  ret <2 x i16> %r
}

; CHECK:      [[F32toU8v2]] = OpFunction [[U8v2]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F32v2]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToU [[U8v2]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x i8> @f32tou8v2(<2 x float> %a) {
  %r = fptoui <2 x float> %a to <2 x i8>
  ret <2 x i8> %r
}

; CHECK:      [[F16toU32v2]] = OpFunction [[U32v2]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F16v2]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToU [[U32v2]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x i32> @f16tou32v2(<2 x half> %a) {
  %r = fptoui <2 x half> %a to <2 x i32>
  ret <2 x i32> %r
}

; CHECK:      [[F16toU16v2]] = OpFunction [[U16v2]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F16v2]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToU [[U16v2]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x i16> @f16tou16v2(<2 x half> %a) {
  %r = fptoui <2 x half> %a to <2 x i16>
  ret <2 x i16> %r
}

; CHECK:      [[F16toU8v2]] = OpFunction [[U8v2]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[F16v2]]
; CHECK:      OpLabel
; CHECK:      [[R:%.*]] = OpConvertFToU [[U8v2]] [[A]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x i8> @f16tou8v2(<2 x half> %a) {
  %r = fptoui <2 x half> %a to <2 x i8>
  ret <2 x i8> %r
}
