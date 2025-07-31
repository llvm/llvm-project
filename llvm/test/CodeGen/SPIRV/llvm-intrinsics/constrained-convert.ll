; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#sf:]] "conv"
; CHECK-DAG: OpName %[[#uf:]] "conv1"
; CHECK-DAG: OpName %[[#fs:]] "conv2"
; CHECK-DAG: OpName %[[#fu:]] "conv3"
; CHECK-DAG: OpName %[[#fe:]] "conv4"
; CHECK-DAG: OpName %[[#ft:]] "conv5"

; CHECK-DAG: OpDecorate %[[#sf]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#uf]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#ft]] FPRoundingMode RTP

; CHECK-NOT: OpDecorate %[[#fs]] FPRoundingMode
; CHECK-NOT: OpDecorate %[[#fu]] FPRoundingMode
; CHECK-NOT: OpDecorate %[[#fe]] FPRoundingMode

; CHECK: %[[#sf]] = OpConvertSToF
; CHECK: %[[#uf]] = OpConvertUToF
; CHECK: %[[#fs]] = OpConvertFToS
; CHECK: %[[#fu]] = OpConvertFToU
; CHECK: %[[#fe]] = OpFConvert
; CHECK: %[[#ft]] = OpFConvert

; Function Attrs: norecurse nounwind strictfp
define dso_local spir_kernel void @test(float %a, i32 %in, i32 %ui) {
entry:
  %conv = tail call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %in, metadata !"round.tonearest", metadata !"fpexcept.strict") #2
  %conv1 = tail call float @llvm.experimental.constrained.uitofp.f32.i32(i32 %ui, metadata !"round.towardzero", metadata !"fpexcept.ignore") #2
  %conv2 = tail call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %conv1, metadata !"fpexcept.ignore") #2
  %conv3 = tail call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %conv1, metadata !"fpexcept.ignore") #2
  %conv4 = tail call double @llvm.experimental.constrained.fpext.f64.f32(float %conv1, metadata !"fpexcept.ignore") #2
  %conv5 = tail call float @llvm.experimental.constrained.fptrunc.f32.f64(double %conv4, metadata !"round.upward", metadata !"fpexcept.ignore") #2
  ret void
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare float @llvm.experimental.constrained.sitofp.f32.i32(i32, metadata, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare float @llvm.experimental.constrained.uitofp.f32.i32(i32, metadata, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare i32 @llvm.experimental.constrained.fptosi.i32.f32(float, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare i32 @llvm.experimental.constrained.fptoui.i32.f32(float, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata) #1
