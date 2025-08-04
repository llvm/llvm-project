; RUN: llc -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#sf:]] "conv"
; CHECK-DAG: OpName %[[#sf1:]] "conv1"
; CHECK-DAG: OpName %[[#sf2:]] "conv2"
; CHECK-DAG: OpName %[[#sf3:]] "conv3"
; CHECK-DAG: OpName %[[#sf4:]] "conv4"
; CHECK-DAG: OpName %[[#uf1:]] "conv5"
; CHECK-DAG: OpName %[[#uf2:]] "conv6"
; CHECK-DAG: OpName %[[#uf3:]] "conv7"
; CHECK-DAG: OpName %[[#uf4:]] "conv8"
; CHECK-DAG: OpName %[[#uf5:]] "conv9"
; CHECK-DAG: OpName %[[#fs1:]] "conv10"
; CHECK-DAG: OpName %[[#fs2:]] "conv11"
; CHECK-DAG: OpName %[[#fs3:]] "conv12"
; CHECK-DAG: OpName %[[#fs4:]] "conv13"
; CHECK-DAG: OpName %[[#fs5:]] "conv14"
; CHECK-DAG: OpName %[[#fu1:]] "conv15"
; CHECK-DAG: OpName %[[#fu2:]] "conv16"
; CHECK-DAG: OpName %[[#fu3:]] "conv17"
; CHECK-DAG: OpName %[[#fu4:]] "conv18"
; CHECK-DAG: OpName %[[#fu5:]] "conv19"
; CHECK-DAG: OpName %[[#fe1:]] "conv20"
; CHECK-DAG: OpName %[[#fe2:]] "conv21"
; CHECK-DAG: OpName %[[#ft1:]] "conv22"
; CHECK-DAG: OpName %[[#ft2:]] "conv23"

; CHECK-DAG: OpConvertSToF %[[#]] %[[#]]
; CHECK-DAG: OpConvertSToF %[[#]] %[[#]]
; CHECK-DAG: OpConvertSToF %[[#]] %[[#]]
; CHECK-DAG: OpConvertSToF %[[#]] %[[#]]
; CHECK-DAG: OpConvertSToF %[[#]] %[[#]]
; CHECK-DAG: OpConvertUToF %[[#]] %[[#]]
; CHECK-DAG: OpConvertUToF %[[#]] %[[#]]
; CHECK-DAG: OpConvertUToF %[[#]] %[[#]]
; CHECK-DAG: OpConvertUToF %[[#]] %[[#]]
; CHECK-DAG: OpConvertUToF %[[#]] %[[#]]
; CHECK-DAG: OpConvertFToS %[[#]] %[[#]]
; CHECK-DAG: OpConvertFToS %[[#]] %[[#]]
; CHECK-DAG: OpConvertFToS %[[#]] %[[#]]
; CHECK-DAG: OpConvertFToS %[[#]] %[[#]]
; CHECK-DAG: OpConvertFToS %[[#]] %[[#]]
; CHECK-DAG: OpFConvert %[[#]] %[[#]]
; CHECK-DAG: OpFConvert %[[#]] %[[#]]
; CHECK-DAG: OpFConvert %[[#]] %[[#]]
; CHECK-DAG: OpFConvert %[[#]] %[[#]]

; CHECK-DAG: OpDecorate %[[#sf]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#sf1]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#sf2]] FPRoundingMode RTP
; CHECK-DAG: OpDecorate %[[#sf3]] FPRoundingMode RTN
; CHECK-DAG: OpDecorate %[[#sf4]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#uf1]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#uf2]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#uf3]] FPRoundingMode RTP
; CHECK-DAG: OpDecorate %[[#uf4]] FPRoundingMode RTN
; CHECK-DAG: OpDecorate %[[#uf5]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#ft1]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#ft2]] FPRoundingMode RTE

define dso_local spir_kernel void @test1(i32 %in) {
entry:
    %conv = tail call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %in, metadata !"round.tonearest", metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test2(i16 %in) {
entry:
    %conv1 = tail call float @llvm.experimental.constrained.sitofp.f32.i16(i16 %in, metadata !"round.towardzero", metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test3(i16 %in) {
entry:
    %conv2 = tail call double @llvm.experimental.constrained.sitofp.f64.i16(i16 %in, metadata !"round.upward", metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test4(i16 %in) {
entry:
    %conv3 = tail call double @llvm.experimental.constrained.sitofp.f64.i32(i16 %in, metadata !"round.downward", metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test5(<4 x i16> %in) {
entry:
    %conv4 = tail call <4 x double > @llvm.experimental.constrained.sitofp.v4f64.v4i32(<4 x i16> %in, metadata !"round.tonearest", metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test6(i32 %in) {
entry:
    %conv5 = tail call float @llvm.experimental.constrained.uitofp.f32.i32(i32 %in, metadata !"round.tonearest", metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test7(i32 %in) {
entry:
    %conv6 = tail call double @llvm.experimental.constrained.uitofp.f64.i32(i32 %in, metadata !"round.towardzero", metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test8(i16 %in) {
entry:
    %conv7 = tail call float @llvm.experimental.constrained.uitofp.f32.i16(i16 %in, metadata !"round.upward", metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test9(i16 %in) {
entry:
    %conv8 = tail call double @llvm.experimental.constrained.uitofp.f64.i16(i16 %in, metadata !"round.downward", metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test10(<4 x i32> %in) {
entry:
    %conv9 = tail call <4 x float> @llvm.experimental.constrained.uitofp.v4f32.v4i32(<4 x i32> %in, metadata !"round.tonearest", metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test11(float %in) {
entry:
    %conv10 = tail call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %in, metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test12(double %in) {
entry:
    %conv11 = tail call i32 @llvm.experimental.constrained.fptosi.i32.f64(double %in, metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test13(float %in) {
entry:
    %conv12 = tail call i16 @llvm.experimental.constrained.fptosi.i16.f64(float %in, metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test14(double %in) {
entry:
    %conv13 = tail call i16 @llvm.experimental.constrained.fptosi.i16.f64(double %in, metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test15(<4 x double> %in) {
entry:
    %conv14 = tail call <4 x i16> @llvm.experimental.constrained.fptosi.v4i16.v4f64(<4 x double> %in, metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test16(float %in) {
entry:
    %conv15 = tail call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %in, metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test17(double %in) {
entry:
    %conv16 = tail call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %in, metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test18(float %in) {
entry:
    %conv17 = tail call i16 @llvm.experimental.constrained.fptoui.i16.f32(float %in, metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test19(double %in) {
entry:
    %conv18 = tail call i16 @llvm.experimental.constrained.fptoui.i16.f64(double %in, metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test20( <4 x double> %in) {
entry:
    %conv19 = tail call <4 x i32> @llvm.experimental.constrained.fptoui.v4i32.v4f64(<4 x double> %in, metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test21(float %in) {
entry:
    %conv20 = tail call double @llvm.experimental.constrained.fpext.f64.f32(float %in,  metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test22(<4 x float> %in) {
entry:
    %conv21 = tail call <4 x double> @llvm.experimental.constrained.fpext.v4f64.v4f32( <4 x float> %in,  metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test23(<4 x double> %in) {
entry:
    %conv22 = tail call <4 x float> @llvm.experimental.constrained.fptrunc.v4f32.v4f64( <4 x double> %in,metadata !"round.towardzero",  metadata !"fpexcept.strict") 
    ret void
}
define dso_local spir_kernel void @test24(double %in) {
entry:
    %conv23 = tail call float @llvm.experimental.constrained.fptrunc.f32.f64( double %in, metadata !"round.tonearest",  metadata !"fpexcept.strict") 
    ret void
}
