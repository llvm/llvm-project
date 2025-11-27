; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#r1:]] "r1"
; CHECK-DAG: OpName %[[#r2:]] "r2"
; CHECK-DAG: OpName %[[#r3:]] "r3"
; CHECK-DAG: OpName %[[#r4:]] "r4"
; CHECK-DAG: OpName %[[#r5:]] "r5"
; CHECK-DAG: OpName %[[#r6:]] "r6"

; CHECK-NOT: OpDecorate %[[#r5]] FPRoundingMode
; CHECK-NOT: OpDecorate %[[#r6]] FPRoundingMode

; CHECK-DAG: OpDecorate %[[#r1]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#r2]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#r4]] FPRoundingMode RTN
; CHECK-DAG: OpDecorate %[[#r3]] FPRoundingMode RTP

; CHECK: OpFAdd %[[#]] %[[#]]
; CHECK: OpFDiv %[[#]] %[[#]]
; CHECK: OpFSub %[[#]] %[[#]]
; CHECK: OpFMul %[[#]] %[[#]]
; CHECK: OpExtInst %[[#]] %[[#]] fma %[[#]] %[[#]] %[[#]]
; CHECK: OpFRem

@G_r1 = global float 0.0
@G_r2 = global float 0.0
@G_r3 = global float 0.0
@G_r4 = global float 0.0
@G_r5 = global float 0.0
@G_r6 = global float 0.0

; Function Attrs: norecurse nounwind strictfp
define dso_local spir_kernel void @test(float %a, i32 %in, i32 %ui) {
entry:
  %r1 = tail call float @llvm.experimental.constrained.fadd.f32(float %a, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  store float %r1, ptr @G_r1
  %r2 = tail call float @llvm.experimental.constrained.fdiv.f32(float %a, float %a, metadata !"round.towardzero", metadata !"fpexcept.strict")
  store float %r2, ptr @G_r2
  %r3 = tail call float @llvm.experimental.constrained.fsub.f32(float %a, float %a, metadata !"round.upward", metadata !"fpexcept.strict")
  store float %r3, ptr @G_r3
  %r4 = tail call float @llvm.experimental.constrained.fmul.f32(float %a, float %a, metadata !"round.downward", metadata !"fpexcept.strict")
  store float %r4, ptr @G_r4
  %r5 = tail call float @llvm.experimental.constrained.fma.f32(float %a, float %a, float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  store float %r5, ptr @G_r5
  %r6 = tail call float @llvm.experimental.constrained.frem.f32(float %a, float %a, metadata !"round.dynamic", metadata !"fpexcept.strict")
  store float %r6, ptr @G_r6
  ret void
}

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fmuladd.f32(float, float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
declare float @llvm.experimental.constrained.frem.f32(float, float, metadata, metadata)
