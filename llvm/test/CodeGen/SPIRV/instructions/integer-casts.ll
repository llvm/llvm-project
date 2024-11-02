; RUN: llc %s -o - | FileCheck %s

target triple = "spirv32-unknown-unknown"

; CHECK-DAG: OpName [[TRUNC32_16:%.*]] "i32toi16"
; CHECK-DAG: OpName [[TRUNC32_8:%.*]] "i32toi8"
; CHECK-DAG: OpName [[TRUNC16_8:%.*]] "i16toi8"
; CHECK-DAG: OpName [[SEXT8_32:%.*]] "s8tos32"
; CHECK-DAG: OpName [[SEXT8_16:%.*]] "s8tos16"
; CHECK-DAG: OpName [[SEXT16_32:%.*]] "s16tos32"
; CHECK-DAG: OpName [[ZEXT8_32:%.*]] "u8tou32"
; CHECK-DAG: OpName [[ZEXT8_16:%.*]] "u8tou16"
; CHECK-DAG: OpName [[ZEXT16_32:%.*]] "u16tou32"

; CHECK-DAG: OpName [[TRUNC32_16v4:%.*]] "i32toi16v4"
; CHECK-DAG: OpName [[TRUNC32_8v4:%.*]] "i32toi8v4"
; CHECK-DAG: OpName [[TRUNC16_8v4:%.*]] "i16toi8v4"
; CHECK-DAG: OpName [[SEXT8_32v4:%.*]] "s8tos32v4"
; CHECK-DAG: OpName [[SEXT8_16v4:%.*]] "s8tos16v4"
; CHECK-DAG: OpName [[SEXT16_32v4:%.*]] "s16tos32v4"
; CHECK-DAG: OpName [[ZEXT8_32v4:%.*]] "u8tou32v4"
; CHECK-DAG: OpName [[ZEXT8_16v4:%.*]] "u8tou16v4"
; CHECK-DAG: OpName [[ZEXT16_32v4:%.*]] "u16tou32v4"

; CHECK-DAG: [[U32:%.*]] = OpTypeInt 32 0
; CHECK-DAG: [[U16:%.*]] = OpTypeInt 16 0
; CHECK-DAG: [[U8:%.*]] = OpTypeInt 8 0
; CHECK-DAG: [[U32v4:%.*]] = OpTypeVector [[U32]] 4
; CHECK-DAG: [[U16v4:%.*]] = OpTypeVector [[U16]] 4
; CHECK-DAG: [[U8v4:%.*]] = OpTypeVector [[U8]] 4


; CHECK: [[TRUNC32_16]] = OpFunction [[U16]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U16]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i16 @i32toi16(i32 %a) {
    %r = trunc i32 %a to i16
    ret i16 %r
}

; CHECK: [[TRUNC32_8]] = OpFunction [[U8]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U8]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i8 @i32toi8(i32 %a) {
    %r = trunc i32 %a to i8
    ret i8 %r
}

; CHECK: [[TRUNC16_8]] = OpFunction [[U8]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U8]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i8 @i16toi8(i16 %a) {
    %r = trunc i16 %a to i8
    ret i8 %r
}


; CHECK: [[SEXT8_32]] = OpFunction [[U32]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpSConvert [[U32]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @s8tos32(i8 %a) {
  %r = sext i8 %a to i32
  ret i32 %r
}

; CHECK: [[SEXT8_16]] = OpFunction [[U16]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpSConvert [[U16]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i16 @s8tos16(i8 %a) {
  %r = sext i8 %a to i16
  ret i16 %r
}

; CHECK: [[SEXT16_32]] = OpFunction [[U32]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpSConvert [[U32]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @s16tos32(i16 %a) {
  %r = sext i16 %a to i32
  ret i32 %r
}

; CHECK: [[ZEXT8_32]] = OpFunction [[U32]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U32]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @u8tou32(i8 %a) {
  %r = zext i8 %a to i32
  ret i32 %r
}

; CHECK: [[ZEXT8_16]] = OpFunction [[U16]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U16]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i16 @u8tou16(i8 %a) {
  %r = zext i8 %a to i16
  ret i16 %r
}

; CHECK: [[ZEXT16_32]] = OpFunction [[U32]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U32]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @u16tou32(i16 %a) {
  %r = zext i16 %a to i32
  ret i32 %r
}

; CHECK: [[TRUNC32_16v4]] = OpFunction [[U16v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32v4]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U16v4]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i16> @i32toi16v4(<4 x i32> %a) {
    %r = trunc <4 x i32> %a to <4 x i16>
    ret <4 x i16> %r
}

; CHECK: [[TRUNC32_8v4]] = OpFunction [[U8v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U32v4]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U8v4]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i8> @i32toi8v4(<4 x i32> %a) {
    %r = trunc <4 x i32> %a to <4 x i8>
    ret <4 x i8> %r
}

; CHECK: [[TRUNC16_8v4]] = OpFunction [[U8v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16v4]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U8v4]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i8> @i16toi8v4(<4 x i16> %a) {
    %r = trunc <4 x i16> %a to <4 x i8>
    ret <4 x i8> %r
}


; CHECK: [[SEXT8_32v4]] = OpFunction [[U32v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v4]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpSConvert [[U32v4]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i32>  @s8tos32v4(<4 x i8> %a) {
  %r = sext <4 x i8> %a to <4 x i32>
  ret <4 x i32>  %r
}

; CHECK: [[SEXT8_16v4]] = OpFunction [[U16v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v4]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpSConvert [[U16v4]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i16> @s8tos16v4(<4 x i8> %a) {
  %r = sext <4 x i8> %a to <4 x i16>
  ret <4 x i16> %r
}

; CHECK: [[SEXT16_32v4]] = OpFunction [[U32v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16v4]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpSConvert [[U32v4]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i32>  @s16tos32v4(<4 x i16> %a) {
  %r = sext <4 x i16> %a to <4 x i32>
  ret <4 x i32>  %r
}

; CHECK: [[ZEXT8_32v4]] = OpFunction [[U32v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v4]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U32v4]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i32>  @u8tou32v4(<4 x i8> %a) {
  %r = zext <4 x i8> %a to <4 x i32>
  ret <4 x i32>  %r
}

; CHECK: [[ZEXT8_16v4]] = OpFunction [[U16v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U8v4]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U16v4]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i16> @u8tou16v4(<4 x i8> %a) {
  %r = zext <4 x i8> %a to <4 x i16>
  ret <4 x i16> %r
}

; CHECK: [[ZEXT16_32v4]] = OpFunction [[U32v4]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[U16v4]]
; CHECK: OpLabel
; CHECK: [[R:%.*]] = OpUConvert [[U32v4]] [[A]]
; CHECK: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <4 x i32>  @u16tou32v4(<4 x i16> %a) {
  %r = zext <4 x i16> %a to <4 x i32>
  ret <4 x i32>  %r
}
