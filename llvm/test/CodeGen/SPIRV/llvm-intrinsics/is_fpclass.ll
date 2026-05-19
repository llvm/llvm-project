; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#BoolTy:]] = OpTypeBool
; CHECK-DAG: %[[#FP32Ty:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FP64Ty:]] = OpTypeFloat 64
; CHECK-DAG: %[[#FP16Ty:]] = OpTypeFloat 16
; CHECK-DAG: %[[#I32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64Ty:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#I16Ty:]] = OpTypeInt 16 0

; CHECK-DAG: %[[#V4I32Ty:]] = OpTypeVector %[[#I32Ty]] 4
; CHECK-DAG: %[[#V4FP32Ty:]] = OpTypeVector %[[#FP32Ty]] 4
; CHECK-DAG: %[[#V4BoolTy:]] = OpTypeVector %[[#BoolTy]] 4

; CHECK-DAG: %[[#MaxExpMinus1:]] = OpConstant %[[#I32Ty]] 2130706432
; CHECK-DAG: %[[#ExpLSB:]] = OpConstant %[[#I32Ty]] 8388608
; CHECK-DAG: %[[#True:]] = OpConstantTrue %[[#BoolTy]]
; CHECK-DAG: %[[#False:]] = OpConstantFalse %[[#BoolTy]]
; CHECK-DAG: %[[#ValueMask:]] = OpConstant %[[#I32Ty]] 2147483647
; CHECK-DAG: %[[#InfWithQnanBit:]] = OpConstant %[[#I32Ty]] 2143289344
; CHECK-DAG: %[[#Inf:]] = OpConstant %[[#I32Ty]] 2139095040
; CHECK-DAG: %[[#NegInf:]] = OpConstant %[[#I32Ty]] 4286578688
; CHECK-DAG: %[[#One:]] = OpConstant %[[#I32Ty]] 1
; CHECK-DAG: %[[#Zero:]] = OpConstantNull %[[#I32Ty]]
; CHECK-DAG: %[[#AllOneMantissa:]] = OpConstant %[[#I32Ty]] 8388607
; CHECK-DAG: %[[#SignBit:]] = OpConstant %[[#I32Ty]] 2147483648

; CHECK-DAG: %[[#ValueMaskFP64:]] = OpConstant %[[#I64Ty]] 9223372036854775807
; CHECK-DAG: %[[#InfFP64:]] = OpConstant %[[#I64Ty]] 9218868437227405312
; CHECK-DAG: %[[#NegInfFP64:]] = OpConstant %[[#I64Ty]] 18442240474082181120

; CHECK-DAG: %[[#FalseV4:]] = OpConstantComposite %[[#V4BoolTy]] %[[#False]] %[[#False]] %[[#False]] %[[#False]]
; CHECK-DAG: %[[#ValueMaskV4:]] = OpConstantComposite %[[#V4I32Ty]] %[[#ValueMask]] %[[#ValueMask]] %[[#ValueMask]] %[[#ValueMask]]
; CHECK-DAG: %[[#InfV4:]] = OpConstantComposite %[[#V4I32Ty]] %[[#Inf]] %[[#Inf]] %[[#Inf]] %[[#Inf]]
; CHECK-DAG: %[[#InfWithQnanBitV4:]] = OpConstantComposite %[[#V4I32Ty]] %[[#InfWithQnanBit]] %[[#InfWithQnanBit]] %[[#InfWithQnanBit]] %[[#InfWithQnanBit]]
; CHECK-DAG: %[[#ValueMaskFP16:]] = OpConstant %[[#I16Ty]] 32767
; CHECK-DAG: %[[#InfFP16:]] = OpConstant %[[#I16Ty]] 31744
; CHECK-DAG: %[[#NegInfFP16:]] = OpConstant %[[#I16Ty]] 64512

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: OpReturnValue %[[#False]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_0_none(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 0)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpUGreaterThan %[[#BoolTy]] %[[#T1]] %[[#Inf]]
; CHECK: %[[#T3:]] = OpULessThan %[[#BoolTy]] %[[#T1]] %[[#InfWithQnanBit]]
; CHECK: %[[#T4:]] = OpLogicalAnd %[[#BoolTy]] %[[#T2]] %[[#T3]]
; CHECK: %[[#T5:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T4]]
; CHECK: OpReturnValue %[[#T5]]
; CHECK: OpFunctionEnd

define i1 @isfpclass_1_issnan(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 1)
  ret i1 %v
}

; CHECK: OpFunction %[[#V4BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#V4FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#V4I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#V4I32Ty]] %[[#T0]] %[[#ValueMaskV4]]
; CHECK: %[[#T2:]] = OpUGreaterThan %[[#V4BoolTy]] %[[#T1]] %[[#InfV4]]
; CHECK: %[[#T3:]] = OpULessThan %[[#V4BoolTy]] %[[#T1]] %[[#InfWithQnanBitV4]]
; CHECK: %[[#T4:]] = OpLogicalAnd %[[#V4BoolTy]] %[[#T2]] %[[#T3]]
; CHECK: %[[#T5:]] = OpLogicalOr %[[#V4BoolTy]] %[[#FalseV4]] %[[#T4]]
; CHECK: OpReturnValue %[[#T5]]
; CHECK: OpFunctionEnd

define <4 x i1> @isfpclass_1_issnan_v4f32(<4 x float> %a) {
  %v = call <4 x i1> @llvm.is.fpclass.v4f32(<4 x float> %a, i32 1)
  ret <4 x i1> %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpUGreaterThanEqual %[[#BoolTy]] %[[#T1]] %[[#InfWithQnanBit]]
; CHECK: %[[#T3:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T2]]
; CHECK: OpReturnValue %[[#T3]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_1_isqnan(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 2)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpUGreaterThan %[[#BoolTy]] %[[#T1]] %[[#Inf]]
; CHECK: %[[#T3:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T2]]
; CHECK: OpReturnValue %[[#T3]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_1_isnan(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 3)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpIEqual %[[#BoolTy]] %[[#T0]] %[[#Inf]]
; CHECK: %[[#T2:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T1]]
; CHECK: OpReturnValue %[[#T2]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_1_ispinf(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 512)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpIEqual %[[#BoolTy]] %[[#T0]] %[[#NegInf]]
; CHECK: %[[#T2:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T1]]
; CHECK: OpReturnValue %[[#T2]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_1_isninf(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 4)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpIEqual %[[#BoolTy]] %[[#T1]] %[[#Inf]]
; CHECK: %[[#T3:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T2]]
; CHECK: OpReturnValue %[[#T3]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_1_isinf(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 516)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpINotEqual %[[#BoolTy]] %[[#T0]] %[[#T1]]
; CHECK: %[[#T3:]] = OpISub %[[#I32Ty]] %[[#T1]] %[[#ExpLSB]]
; CHECK: %[[#T4:]] = OpULessThan %[[#BoolTy]] %[[#T3]] %[[#MaxExpMinus1]]
; CHECK: %[[#T5:]] = OpLogicalNotEqual %[[#BoolTy]] %[[#T2]] %[[#True]]
; CHECK: %[[#T6:]] = OpLogicalAnd %[[#BoolTy]] %[[#T4]] %[[#T5]]
; CHECK: %[[#T7:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T6]]
; CHECK: OpReturnValue %[[#T7]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_isposnormal(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 256)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpINotEqual %[[#BoolTy]] %[[#T0]] %[[#T1]]
; CHECK: %[[#T3:]] = OpISub %[[#I32Ty]] %[[#T1]] %[[#ExpLSB]]
; CHECK: %[[#T4:]] = OpULessThan %[[#BoolTy]] %[[#T3]] %[[#MaxExpMinus1]]
; CHECK: %[[#T5:]] = OpLogicalAnd %[[#BoolTy]] %[[#T4]] %[[#T2]]
; CHECK: %[[#T6:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T5]]
; CHECK: OpReturnValue %[[#T6]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_isnegnormal(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 8)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpISub %[[#I32Ty]] %[[#T1]] %[[#ExpLSB]]
; CHECK: %[[#T3:]] = OpULessThan %[[#BoolTy]] %[[#T2]] %[[#MaxExpMinus1]]
; CHECK: %[[#T4:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T3]]
; CHECK: OpReturnValue %[[#T4]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_isnormal(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 264)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpUGreaterThan %[[#BoolTy]] %[[#T1]] %[[#Inf]]
; CHECK: %[[#T3:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T2]]
; CHECK: %[[#T4:]] = OpISub %[[#I32Ty]] %[[#T1]] %[[#ExpLSB]]
; CHECK: %[[#T5:]] = OpULessThan %[[#BoolTy]] %[[#T4]] %[[#MaxExpMinus1]]
; CHECK: %[[#T6:]] = OpLogicalOr %[[#BoolTy]] %[[#T3]] %[[#T5]]
; CHECK: OpReturnValue %[[#T6]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_1_isnan_or_normal(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 267)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpISub %[[#I32Ty]] %[[#T0]] %[[#One]]
; CHECK: %[[#T2:]] = OpULessThan %[[#BoolTy]] %[[#T1]] %[[#AllOneMantissa]]
; CHECK: %[[#T3:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T2]]
; CHECK: OpReturnValue %[[#T3]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_ispsubnormal(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 128)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpINotEqual %[[#BoolTy]] %[[#T0]] %[[#T1]]
; CHECK: %[[#T3:]] = OpISub %[[#I32Ty]] %[[#T1]] %[[#One]]
; CHECK: %[[#T4:]] = OpULessThan %[[#BoolTy]] %[[#T3]] %[[#AllOneMantissa]]
; CHECK: %[[#T5:]] = OpLogicalAnd %[[#BoolTy]] %[[#T4]] %[[#T2]]
; CHECK: %[[#T6:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T5]]
; CHECK: OpReturnValue %[[#T6]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_isnsubnormal(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 16)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpISub %[[#I32Ty]] %[[#T1]] %[[#One]]
; CHECK: %[[#T3:]] = OpULessThan %[[#BoolTy]] %[[#T2]] %[[#AllOneMantissa]]
; CHECK: %[[#T4:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T3]]
; CHECK: OpReturnValue %[[#T4]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_issubnormal(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 144)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpIEqual %[[#BoolTy]] %[[#T0]] %[[#Zero]]
; CHECK: %[[#T2:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T1]]
; CHECK: OpReturnValue %[[#T2]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_ispzero(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 64)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpIEqual %[[#BoolTy]] %[[#T0]] %[[#SignBit]]
; CHECK: %[[#T2:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T1]]
; CHECK: OpReturnValue %[[#T2]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_isnzero(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 32)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpIEqual %[[#BoolTy]] %[[#T1]] %[[#Zero]]
; CHECK: %[[#T3:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T2]]
; CHECK: OpReturnValue %[[#T3]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_iszero(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 96)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpULessThan %[[#BoolTy]] %[[#T0]] %[[#Inf]]
; CHECK: %[[#T2:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T1]]
; CHECK: OpReturnValue %[[#T2]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_ispfinite(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 448)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpINotEqual %[[#BoolTy]] %[[#T0]] %[[#T1]]
; CHECK: %[[#T3:]] = OpULessThan %[[#BoolTy]] %[[#T1]] %[[#Inf]]
; CHECK: %[[#T4:]] = OpLogicalAnd %[[#BoolTy]] %[[#T3]] %[[#T2]]
; CHECK: %[[#T5:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T4]]
; CHECK: OpReturnValue %[[#T5]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_isnfinite(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 56)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpULessThan %[[#BoolTy]] %[[#T1]] %[[#Inf]]
; CHECK: %[[#T3:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T2]]
; CHECK: OpReturnValue %[[#T3]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_isfinite(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 504)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpULessThan %[[#BoolTy]] %[[#T0]] %[[#Inf]]
; CHECK: %[[#T2:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T1]]
; CHECK: %[[#T3:]] = OpIEqual %[[#BoolTy]] %[[#T0]] %[[#Inf]]
; CHECK: %[[#T4:]] = OpLogicalOr %[[#BoolTy]] %[[#T2]] %[[#T3]]
; CHECK: OpReturnValue %[[#T4]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_ispositive(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 960)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I32Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I32Ty]] %[[#T0]] %[[#ValueMask]]
; CHECK: %[[#T2:]] = OpINotEqual %[[#BoolTy]] %[[#T0]] %[[#T1]]
; CHECK: %[[#T3:]] = OpULessThan %[[#BoolTy]] %[[#T1]] %[[#Inf]]
; CHECK: %[[#T4:]] = OpLogicalAnd %[[#BoolTy]] %[[#T3]] %[[#T2]]
; CHECK: %[[#T5:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T4]]
; CHECK: %[[#T6:]] = OpIEqual %[[#BoolTy]] %[[#T0]] %[[#NegInf]]
; CHECK: %[[#T7:]] = OpLogicalOr %[[#BoolTy]] %[[#T5]] %[[#T6]]
; CHECK: OpReturnValue %[[#T7]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_isnegative(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 60)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: OpReturnValue %[[#True]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_all(float %a) {
  %v = call i1 @llvm.is.fpclass.f32(float %a, i32 1023)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP64Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I64Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I64Ty]] %[[#T0]] %[[#ValueMaskFP64]]
; CHECK: %[[#T2:]] = OpINotEqual %[[#BoolTy]] %[[#T0]] %[[#T1]]
; CHECK: %[[#T3:]] = OpULessThan %[[#BoolTy]] %[[#T1]] %[[#InfFP64]]
; CHECK: %[[#T4:]] = OpLogicalAnd %[[#BoolTy]] %[[#T3]] %[[#T2]]
; CHECK: %[[#T5:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T4]]
; CHECK: %[[#T6:]] = OpIEqual %[[#BoolTy]] %[[#T0]] %[[#NegInfFP64]]
; CHECK: %[[#T7:]] = OpLogicalOr %[[#BoolTy]] %[[#T5]] %[[#T6]]
; CHECK: OpReturnValue %[[#T7]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_f64_isnegative(double %a) {
  %v = call i1 @llvm.is.fpclass.f64(double %a, i32 60)
  ret i1 %v
}

; CHECK: OpFunction %[[#BoolTy]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#FP16Ty]]
; CHECK: %[[#T0:]] = OpBitcast %[[#I16Ty]] %[[#A]]
; CHECK: %[[#T1:]] = OpBitwiseAnd %[[#I16Ty]] %[[#T0]] %[[#ValueMaskFP16]]
; CHECK: %[[#T2:]] = OpINotEqual %[[#BoolTy]] %[[#T0]] %[[#T1]]
; CHECK: %[[#T3:]] = OpULessThan %[[#BoolTy]] %[[#T1]] %[[#InfFP16]]
; CHECK: %[[#T4:]] = OpLogicalAnd %[[#BoolTy]] %[[#T3]] %[[#T2]]
; CHECK: %[[#T5:]] = OpLogicalOr %[[#BoolTy]] %[[#False]] %[[#T4]]
; CHECK: %[[#T6:]] = OpIEqual %[[#BoolTy]] %[[#T0]] %[[#NegInfFP16]]
; CHECK: %[[#T7:]] = OpLogicalOr %[[#BoolTy]] %[[#T5]] %[[#T6]]
; CHECK: OpReturnValue %[[#T7]]
; CHECK: OpFunctionEnd
define i1 @isfpclass_f16_isnegative(half %a) {
  %v = call i1 @llvm.is.fpclass.f16(half %a, i32 60)
  ret i1 %v
}

declare i1 @llvm.is.fpclass.f32(float, i32)
declare <4 x i1> @llvm.is.fpclass.v4f32(<4 x float>, i32)
declare i1 @llvm.is.fpclass.f64(double, i32)
declare i1 @llvm.is.fpclass.f16(half, i32)
