;; Ensure @llvm.fptosi.sat.* and @llvm.fptoui.sat.* intrinsics are translated
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unkown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unkown-unknown %s -o - -filetype=obj | spirv-val %}
 
; CHECK-DAG: OpCapability Kernel
; CHECK-DAG: OpDecorate %[[#SAT1:]] SaturatedConversion
; CHECK-DAG: OpDecorate %[[#SAT2:]] SaturatedConversion
; CHECK-DAG: %[[#INT64TY:]] = OpTypeInt  64 0
; CHECK-DAG: %[[#BOOLTY:]] = OpTypeBool 
; CHECK-DAG: %[[#IS2MIN:]] = OpConstant %[[#INT64TY]]
; CHECK-DAG: %[[#I2SMAX:]] = OpConstant %[[#INT64TY]] 
; CHECK-DAG: %[[#SAT1]] = OpConvertFToS %[[#INT64TY]] %[[#]]
; CHECK-DAG: %[[#SGRES:]] = OpSGreaterThanEqual %[[#BOOLTY]] %[[#SAT1]] %[[#I2SMAX]]
; CHECK-DAG: %[[#SLERES:]] = OpSLessThanEqual %[[#BOOLTY]] %[[#SAT1]] %[[#IS2MIN]]
; CHECK-DAG: %[[#SELRES1:]] = OpSelect %[[#INT64TY]] %[[#SGRES]] %[[#I2SMAX]] %[[#SAT1]]
; CHECK-DAG: %[[#SELRES2:]] = OpSelect %[[#INT64TY]] %[[#SLERES]] %[[#IS2MIN]] %[[#SELRES1]]
 
define spir_kernel void @testfunction_float_to_signed_i2(float %input) {
entry:
    %ptr = alloca i64
   %0 = call i2 @llvm.fptosi.sat.i2.f32(float %input)
   %1 = sext i2 %0 to i64
   store i64 %1, i64* %ptr
   ret void

}
declare i2 @llvm.fptosi.sat.i2.f32(float)


; CHECK-DAG: %[[#I2UMAX:]] = OpConstant %[[#INT64TY]] 3 
; CHECK-DAG: %[[#SAT2]] = OpConvertFToU %[[#INT64TY]] %[[#]] 
; CHECK-DAG: %[[#UGERES:]] = OpUGreaterThanEqual %[[#BOOLTY]] %[[#SAT2]] %[[#I2UMAX]]
; CHECK-DAG: %[[#SELRES1U:]] = OpSelect %[[#INT64TY]] %[[#UGERES]] %[[#I2UMAX]] %[[#SAT2]]

define spir_kernel void @testfunction_float_to_unsigned_i2(float %input) {
entry:
   %ptr = alloca i64
   %0 = call i2 @llvm.fptoui.sat.i2.f32(float %input)
   %1 = zext i2 %0 to i64
   store i64 %1, i64* %ptr
   ret void

}
declare i2 @llvm.fptoui.sat.i2.f32(float)
 
 
