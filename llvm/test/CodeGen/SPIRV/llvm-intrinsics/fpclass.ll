; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: %[[#Int32Ty:]] =  OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#Int64Ty:]] =  OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#Int16Ty:]] =  OpTypeInt  16 0
; CHECK-SPIRV-DAG: %[[#BoolTy:]]  =  OpTypeBool
; CHECK-SPIRV-DAG: %[[#VecBoolTy:]] =  OpTypeVector %[[#BoolTy]] 2
; CHECK-SPIRV-DAG: %[[#Int32VecTy:]] =  OpTypeVector %[[#Int32Ty]] 2
; CHECK-SPIRV-DAG: %[[#Int16VecTy:]] = OpTypeVector %[[#Int16Ty]] 2
; CHECK-SPIRV-DAG: %[[#DoubleTy:]] =  OpTypeFloat 64
; CHECK-SPIRV-DAG: %[[#QNanBitConst:]] =  OpConstant %[[#Int32Ty]] 2143289344
; CHECK-SPIRV-DAG: %[[#MantissaConst:]] =  OpConstant %[[#Int32Ty]] 8388607
; CHECK-SPIRV-DAG: %[[#ZeroConst:]] = OpConstant %[[#Int32Ty]] 0
; CHECK-SPIRV-DAG: %[[#MaskToClearSignBit:]] =  OpConstant %[[#Int32Ty]] 2147483647
; CHECK-SPIRV-DAG: %[[#NegatedZeroConst:]] =  OpConstant %[[#Int32Ty]] 2147483648
; CHECK-SPIRV-DAG: %[[#MantissaConst16:]] = OpConstant %[[#Int16Ty]] 1023
; CHECK-SPIRV-DAG: %[[#ZeroConst64:]] = OpConstant %[[#Int64Ty]] 0
; CHECK-SPIRV-DAG: %[[#OneConst16:]] = OpConstant %[[#Int16Ty]] 1
; CHECK-SPIRV-DAG: %[[#OneConstant:]] = OpConstant %[[#Int64Ty]] 1
; CHECK-SPIRV-DAG: %[[#OneConstVec16:]] = OpConstantComposite %[[#Int16VecTy]] %[[#OneConst16]] %[[#OneConst16]] 
; CHECK-SPIRV-DAG: %[[#QNanBitConst64:]] = OpConstant %[[#Int64Ty]] 9221120237041090560
; CHECK-SPIRV-DAG: %[[#MantissaConst64:]] = OpConstant %[[#Int64Ty]] 4503599627370495
; CHECK-SPIRV-DAG: %[[#QNanBitConst16:]] =  OpConstant %[[#Int16Ty]] 32256
; CHECK-SPIRV-DAG: %[[#QNanBitConstVec16:]] = OpConstantComposite %[[#Int16VecTy:]] %[[#QNanBitConst16]] %[[#QNanBitConst16]]
; CHECK-SPIRV-DAG: %[[#MantissaConstVec16:]] =  OpConstantComposite %[[#Int16VecTy:]] %[[#MantissaConst16]] %[[#MantissaConst16]]
; CHECK-SPIRV-DAG: %[[#ZeroConst16:]] = OpConstant %[[#Int16Ty]] 0
; CHECK-SPIRV-DAG: %[[#zeroConst16Vec2:]] = OpConstantComposite %[[#Int16VecTy]]  %[[#ZeroConst16]] %[[#ZeroConst16]]


; ConstantTrue [[#BoolTy]] [[#True:]]
; ConstantFalse [[#BoolTy]] [[#False:]]

; CHECK-SPIRV-DAG: OpName %[[#NanFunc:]] "test_class_isnan_f32"
; CHECK-SPIRV-DAG: OpName %[[#VecNanFunc:]] "test_class_isnan_v2f32"
; CHECK-SPIRV-DAG: OpName %[[#SNanFunc:]] "test_class_issnan_f32"
; CHECK-SPIRV-DAG: OpName %[[#QNanFunc:]] "test_class_isqnan_f32"
; CHECK-SPIRV-DAG: OpName %[[#InfFunc:]] "test_class_is_inf_f32"
; CHECK-SPIRV-DAG: OpName %[[#InfVecFunc:]] "test_class_is_inf_v2f32"
; CHECK-SPIRV-DAG: OpName %[[#PosInfFunc:]] "test_class_is_pinf_f32"
; CHECK-SPIRV-DAG: OpName %[[#PosInfVecFunc:]] "test_class_is_pinf_v2f32"
; CHECK-SPIRV-DAG: OpName %[[#NegInfFunc:]] "test_class_is_ninf_f32"
; CHECK-SPIRV-DAG: OpName %[[#NegInfVecFunc:]] "test_class_is_ninf_v2f32"
; CHECK-SPIRV-DAG: OpName %[[#NormFunc:]] "test_class_is_normal"
; CHECK-SPIRV-DAG: OpName %[[#PosNormFunc:]] "test_constant_class_pnormal"
; CHECK-SPIRV-DAG: OpName %[[#NegNormFunc:]] "test_constant_class_nnormal"
; CHECK-SPIRV-DAG: OpName %[[#SubnormFunc:]] "test_class_subnormal"
; CHECK-SPIRV-DAG: OpName %[[#PosSubnormFunc:]] "test_class_possubnormal"
; CHECK-SPIRV-DAG: OpName %[[#NegSubnormFunc:]] "test_class_negsubnormal"
; CHECK-SPIRV-DAG: OpName %[[#ZeroFunc:]] "test_class_zero"
; CHECK-SPIRV-DAG: OpName %[[#PosZeroFunc:]] "test_class_poszero"
; CHECK-SPIRV-DAG: OpName %[[#NegZeroFunc:]] "test_class_negzero"
; CHECK-SPIRV-DAG: OpName %[[#NegInfOrNanFunc:]] "test_class_is_ninf_or_nan_f32"
; CHECK-SPIRV-DAG: OpName %[[#ComplexFunc1:]] "test_class_neginf_posnormal_negsubnormal_poszero_snan_f64"
; CHECK-SPIRV-DAG: OpName %[[#ComplexFunc2:]] "test_class_neginf_posnormal_negsubnormal_poszero_snan_v2f16"


; ModuleID = 'fpclass.bc'
source_filename = "fpclass.ll"
target triple = "spir64-unknown-unknown"



; check for nan
define i1 @test_class_isnan_f32(float %x) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: %[[#]] = OpLabel
; CHECK-SPIRV-NEXT: %[[#IsNan:]] = OpIsNan %[[#BoolTy]] %[[#val]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#IsNan]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 3)
  ret i1 %val
}

define <2 x i1> @test_class_isnan_v2f32(<2 x float> %x) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#IsNan:]]  =  OpIsNan %[[#VecBoolTy]] %[[#val]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#IsNan]]
  %val = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %x, i32 3)
  ret <2 x i1> %val
}

; check for snan
define i1 @test_class_issnan_f32(float %x) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#BitCast:]] =  OpBitcast %[[#Int32Ty]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#GECheck:]] =  OpUGreaterThanEqual %[[#]] %[[#BitCast]] %[[#QNanBitConst]]
; CHECK-SPIRV-NEXT: %[[#Not:]] = OpLogicalNot %[[#]] %[[#GECheck]]
; CHECK-SPIRV-NEXT: %[[#IsNan:]] =  OpIsNan %[[#BoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#And:]] =  OpLogicalAnd %[[#]] %[[#IsNan]] %[[#Not]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#And]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 1)
  ret i1 %val
}

; check for qnan
define i1 @test_class_isqnan_f32(float %x) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#BitCast:]] =  OpBitcast %[[#Int32Ty]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#GECheck:]] =  OpUGreaterThanEqual %[[#]] %[[#BitCast]] %[[#QNanBitConst]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#GECheck]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 2)
  ret i1 %val
}

; check for inf
define i1 @test_class_is_inf_f32(float %x) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#IsInf:]] =  OpIsInf %[[#BoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#IsInf]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 516)
  ret i1 %val
}

define <2 x i1> @test_class_is_inf_v2f32(<2 x float> %x) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#IsInf:]] =  OpIsInf %[[#VecBoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#IsInf]]
  %val = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %x, i32 516)
  ret <2 x i1> %val
}

; check for pos inf
define i1 @test_class_is_pinf_f32(float %x) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#IsInf:]] =  OpIsInf %[[#BoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sign:]] =  OpSignBitSet %[[#BoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Not:]] =  OpLogicalNot %[[#BoolTy]] %[[#Sign]]
; CHECK-SPIRV-NEXT: %[[#And:]] =  OpLogicalAnd %[[#BoolTy]] %[[#Not]] %[[#IsInf]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#And]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 512)
  ret i1 %val
}

define <2 x i1> @test_class_is_pinf_v2f32(<2 x float> %x) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#IsInf:]] =  OpIsInf %[[#VecBoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sign:]] =  OpSignBitSet %[[#VecBoolTy]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Not:]] =  OpLogicalNot %[[#VecBoolTy]]  %[[#Sign]]
; CHECK-SPIRV-NEXT: %[[#And:]] =  OpLogicalAnd %[[#VecBoolTy]] %[[#Not]] %[[#IsInf]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#And]]
  %val = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %x, i32 512)
  ret <2 x i1> %val
}

; check for neg inf
define i1 @test_class_is_ninf_f32(float %x) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#IsInf:]] =  OpIsInf %[[#BoolTy]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sign:]] =  OpSignBitSet %[[#BoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#And:]] =  OpLogicalAnd %[[#BoolTy]] %[[#Sign]] %[[#IsInf]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#And]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 4)
  ret i1 %val
}

define <2 x i1> @test_class_is_ninf_v2f32(<2 x float> %x) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#IsInf:]] = OpIsInf %[[#VecBoolTy]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sign:]] = OpSignBitSet %[[#VecBoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#And:]] = OpLogicalAnd %[[#VecBoolTy]]  %[[#Sign]] %[[#IsInf]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#And]]
  %val = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %x, i32 4)
  ret <2 x i1> %val
}

; check for normal
define i1 @test_class_is_normal(float %x) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#IsNormal:]] = OpIsNormal %[[#BoolTy]]  %[[#Val]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#IsNormal]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 264)
  ret i1 %val
}

; check for pos normal
define i1 @test_constant_class_pnormal() {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#IsNormal:]] = OpIsNormal %[[#BoolTy]]  %[[#]]
; CHECK-SPIRV-NEXT: %[[#Sign:]] =  OpSignBitSet %[[#BoolTy]] %[[#]]
; CHECK-SPIRV-NEXT: %[[#Not:]] = OpLogicalNot %[[#BoolTy]]  %[[#Sign]]
; CHECK-SPIRV-NEXT: %[[#And:]] = OpLogicalAnd %[[#BoolTy]] %[[#Not]] %[[#IsNormal]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#And]]
  %val = call i1 @llvm.is.fpclass.f64(double 1.000000e+00, i32 256)
  ret i1 %val
}
; check for neg normal
define i1 @test_constant_class_nnormal() {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#IsNormal:]] =  OpIsNormal %[[#BoolTy]] %[[#]]
; CHECK-SPIRV-NEXT: %[[#Sign:]] = OpSignBitSet %[[#BoolTy]] %[[#]]
; CHECK-SPIRV-NEXT: %[[#And:]] = OpLogicalAnd %[[#BoolTy]] %[[#Sign]] %[[#IsNormal]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#And]]
  %val = call i1 @llvm.is.fpclass.f64(double 1.000000e+00, i32 8)
  ret i1 %val
}

; check for subnormal
define i1 @test_class_subnormal(float %arg) {
; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#BitCast:]] =  OpBitcast %[[#Int32Ty]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sub:]] = OpISub %[[#Int32Ty]]  %[[#BitCast]] %[[#OneConstant:]]
; CHECK-SPIRV-NEXT: %[[#Less:]] = OpULessThan %[[#BoolTy]] %[[#Sub]] %[[#MantissaConst:]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#Less]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 144)
  ret i1 %val
}

; check for pos subnormal
define i1 @test_class_possubnormal(float %arg) {
;; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#BitCast:]] = OpBitcast %[[#Int32Ty]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sub:]] = OpISub %[[#Int32Ty]]  %[[#BitCast]] %[[#OneConstant:]]
; CHECK-SPIRV-NEXT: %[[#Less:]] = OpULessThan %[[#BoolTy]]  %[[#Sub]] %[[#MantissaConst:]]
; CHECK-SPIRV-NEXT: %[[#Sign:]] =  OpSignBitSet %[[#BoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Not:]] = OpLogicalNot %[[#BoolTy]]  %[[#Sign]]
; CHECK-SPIRV-NEXT: %[[#And:]] = OpLogicalAnd %[[#BoolTy]] %[[#Not]] %[[#Less]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#And]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 128)
  ret i1 %val
}

; check for neg subnormal
define i1 @test_class_negsubnormal(float %arg) {
;; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel
; CHECK-SPIRV-NEXT: %[[#BitCast:]] = OpBitcast %[[#Int32Ty]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sub:]] = OpISub %[[#Int32Ty]]  %[[#BitCast]] %[[#OneConstant:]]
; CHECK-SPIRV-NEXT: %[[#Less:]] =  OpULessThan %[[#BoolTy]]  %[[#Sub]] %[[#MantissaConst:]]
; CHECK-SPIRV-NEXT: %[[#Sign:]] = OpSignBitSet %[[#BoolTy]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#And:]] =  OpLogicalAnd %[[#BoolTy]]  %[[#Sign]] %[[#Less]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#And]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 16)
  ret i1 %val
}

; check for zero
define i1 @test_class_zero(float %arg) {
;; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel

; CHECK-SPIRV-NEXT: %[[#BitCast:]] = OpBitcast %[[#Int32Ty]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#BitwiseAndRes:]] = OpBitwiseAnd %[[#Int32Ty]]  %[[#MaskToClearSignBit]] %[[#BitCast]]
; CHECK-SPIRV-NEXT: %[[#EqualPos:]] = OpIEqual %[[#BoolTy]] %[[#BitwiseAndRes]] %[[#ZeroConst]] 
; CHECK-SPIRV-NEXT: ReturnValue %[[#EqualPos]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 96)
  ret i1 %val
}

; check for pos zero
define i1 @test_class_poszero(float %arg) {
;; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel

; CHECK-SPIRV-NEXT: %[[#BitCast:]] = OpBitcast %[[#Int32Ty]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Equal:]] = OpIEqual %[[#BoolTy]]  %[[#ZeroConst]] %[[#BitCast]] 
; CHECK-SPIRV-NEXT: ReturnValue %[[#Equal]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 64)
  ret i1 %val
}

; check for neg zero
define i1 @test_class_negzero(float %arg) {
;; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel

; CHECK-SPIRV-NEXT: %[[#BitCast:]] = OpBitcast %[[#Int32Ty]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Equal:]] = OpIEqual %[[#BoolTy]]  %[[#NegatedZeroConst]] %[[#BitCast]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#Equal]]
  %val = call i1 @llvm.is.fpclass.f32(float %arg, i32 32)
  ret i1 %val
}

; check for neg inf or nan
define i1 @test_class_is_ninf_or_nan_f32(float %x) {
;; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel

; CHECK-SPIRV-NEXT: %[[#IsNan:]] = OpIsNan %[[#BoolTy]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#IsInf:]] = OpIsInf %[[#BoolTy]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sign:]] =  OpSignBitSet %[[#BoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#And:]] = OpLogicalAnd %[[#BoolTy]] %[[#Sign]] %[[#IsInf]]
; CHECK-SPIRV-NEXT: %[[#Or:]] = OpLogicalOr %[[#BoolTy]]  %[[#IsNan]] %[[#And]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#Or]]
  %val = call i1 @llvm.is.fpclass.f32(float %x, i32 7)
  ret i1 %val
}

; check for neg inf, pos normal, neg subnormal pos zero and snan scalar
define i1 @test_class_neginf_posnormal_negsubnormal_poszero_snan_f64(double %arg) {
;; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel

; CHECK-SPIRV-NEXT: %[[#BitCast1:]] =  OpBitcast %[[#Int64Ty]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#GECheck:]] = OpUGreaterThanEqual %[[#BoolTy]]  %[[#BitCast1]] %[[#QNanBitConst64]]
; CHECK-SPIRV-NEXT: %[[#Not1:]] = OpLogicalNot %[[#BoolTy]]  %[[#GECheck]]
; CHECK-SPIRV-NEXT: %[[#IsNan:]] =  OpIsNan %[[#BoolTy]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#And1:]] = OpLogicalAnd %[[#BoolTy]]  %[[#IsNan]] %[[#Not1]]
; CHECK-SPIRV-NEXT: %[[#IsInf:]] = OpIsInf %[[#BoolTy]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sign:]] = OpSignBitSet %[[#BoolTy]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#And2:]] = OpLogicalAnd %[[#BoolTy]]  %[[#Sign]] %[[#IsInf]]
; CHECK-SPIRV-NEXT: %[[#IsNormal:]] =  OpIsNormal %[[#BoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Not2:]] =  OpLogicalNot %[[#BoolTy]] %[[#Sign]]
; CHECK-SPIRV-NEXT: %[[#And3:]] = OpLogicalAnd %[[#BoolTy]] %[[#Not2]] %[[#IsNormal]]
; CHECK-SPIRV-NEXT: %[[#BitCast2:]] = OpBitcast %[[#Int64Ty]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sub:]] = OpISub %[[#Int64Ty]] %[[#BitCast2]] %[[#]]
; CHECK-SPIRV-NEXT: %[[#Less:]] = OpULessThan %[[#BoolTy]] %[[#Sub]] %[[#MantissaConst64]]
; CHECK-SPIRV-NEXT: %[[#And4:]] = OpLogicalAnd %[[#BoolTy]]  %[[#Sign]] %[[#Less]]
; CHECK-SPIRV-NEXT: %[[#BitCast3:]] = OpBitcast %[[#Int64Ty]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Equal:]] = OpIEqual %[[#BoolTy]] %[[#ZeroConst64]] %[[#BitCast3]] 
; CHECK-SPIRV-NEXT: %[[#Or1:]] = OpLogicalOr %[[#BoolTy]]  %[[#And1]] %[[#And2]]

; CHECK-SPIRV-NEXT: %[[#Or2:]] =  OpLogicalOr %[[#BoolTy]] %[[#Or1]] %[[#And3]]
; CHECK-SPIRV-NEXT: %[[#Or3:]] = OpLogicalOr %[[#BoolTy]]  %[[#Or2]] %[[#And4]]
; CHECK-SPIRV-NEXT: %[[#Or4:]] = OpLogicalOr %[[#BoolTy]]  %[[#Or3]] %[[#Equal]]
; CHECK-SPIRV-NEXT: ReturnValue %[[#Or4]]
  %val = call i1 @llvm.is.fpclass.f64(double %arg, i32 341)
  ret i1 %val
}

; check for neg inf, pos normal, neg subnormal pos zero and snan vector
define <2 x i1> @test_class_neginf_posnormal_negsubnormal_poszero_snan_v2f16(<2 x half> %arg) {
;; CHECK-SPIRV: %[[#]] =  OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV-NEXT: %[[#Val:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-NEXT: OpLabel

; CHECK-SPIRV-NEXT: %[[#BitCast1:]] = OpBitcast %[[#Int16VecTy]]  %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#GECheck:]] = OpUGreaterThanEqual %[[#VecBoolTy]] %[[#BitCast1]] %[[#QNanBitConstVec16]]
; CHECK-SPIRV-NEXT: %[[#Not1:]] = OpLogicalNot %[[#VecBoolTy]] %[[#GECheck]]
; CHECK-SPIRV-NEXT: %[[#IsNan:]] = OpIsNan %[[#VecBoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#And1:]] = OpLogicalAnd %[[#VecBoolTy]] %[[#IsNan]] %[[#Not1]]
; CHECK-SPIRV-NEXT: %[[#IsInf:]] = OpIsInf %[[#VecBoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sign:]] = OpSignBitSet %[[#VecBoolTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#And2:]] = OpLogicalAnd %[[#VecBoolTy]] %[[#Sign]] %[[#IsInf]]
; CHECK-SPIRV-NEXT: %[[#IsNormal:]] = OpIsNormal %[[#VecBoolTy]] %[[#Val]]

; CHECK-SPIRV-NEXT: %[[#Not2:]] = OpLogicalNot %[[#VecBoolTy]] %[[#Sign]]
; CHECK-SPIRV-NEXT: %[[#And3:]] = OpLogicalAnd %[[#VecBoolTy]] %[[#Not2]] %[[#IsNormal]]
; CHECK-SPIRV-NEXT: %[[#BitCast2:]] = OpBitcast %[[#Int16VecTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Sub:]] =  OpISub %[[#Int16VecTy]] %[[#BitCast2]] %[[#OneConstVec16]]
; CHECK-SPIRV-NEXT: %[[#Less:]] = OpULessThan %[[#VecBoolTy]] %[[#Sub]] %[[#MantissaConstVec16]]
; CHECK-SPIRV-NEXT: %[[#And4:]] = OpLogicalAnd %[[#VecBoolTy]] %[[#Sign]] %[[#Less]]
; CHECK-SPIRV-NEXT: %[[#BitCast3:]] = OpBitcast %[[#Int16VecTy]] %[[#Val]]
; CHECK-SPIRV-NEXT: %[[#Equal:]] = OpIEqual %[[#VecBoolTy]] %[[#]] %[[#BitCast3]]
; CHECK-SPIRV-NEXT: %[[#Or1:]] = OpLogicalOr %[[#VecBoolTy]] %[[#And1]] %[[#And2]]
; CHECK-SPIRV-NEXT: %[[#Or2:]] =  OpLogicalOr %[[#VecBoolTy]] %[[#Or1]] %[[#And3]]
; CHECK-SPIRV-NEXT: %[[#Or3:]] = OpLogicalOr %[[#VecBoolTy]] %[[#Or2]] %[[#And4]]
; CHECK-SPIRV-NEXT: %[[#Or4:]] = OpLogicalOr %[[#VecBoolTy]] %[[#Or3]] %[[#Equal]]
; CHECK-SPIRV-NEXT: OpReturnValue %[[#Or4]]
  %val = call <2 x i1> @llvm.is.fpclass.v2f16(<2 x half> %arg, i32 341)
  ret <2 x i1> %val
}



declare i1 @llvm.is.fpclass.f32(float, i32 immarg)
declare i1 @llvm.is.fpclass.f64(double, i32 immarg)
declare <2 x i1> @llvm.is.fpclass.v2f32(<2 x float>, i32 immarg)
declare <2 x i1> @llvm.is.fpclass.v2f16(<2 x half>, i32 immarg)
