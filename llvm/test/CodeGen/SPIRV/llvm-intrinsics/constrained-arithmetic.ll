; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; TODO: re-enable validator FPRoundingMode is placed correctly
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt %s -o - | FileCheck %s --check-prefix=CHECK-EXT
; TODO: even with the extension enabled, only FDiv and sqrt instruction can have FPRoundingMode decoration. Re-enable spirv-val once this is fixed.
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt %s -o - -filetype=obj | spirv-val %}

; CHECK-EXT-DAG: OpCapability RoundedDivideSqrtINTEL
; CHECK-EXT-DAG: OpExtension "SPV_INTEL_rounded_divide_sqrt"
; CHECK-EXT-DAG: %[[#e2:]] = OpFDiv %[[#]] %[[#]]
; CHECK-EXT-DAG: %[[#e7:]] = OpExtInst %[[#]] %[[#]] sqrt
; CHECK-EXT-DAG: OpDecorate %[[#e2]] FPRoundingMode RTZ
; CHECK-EXT-DAG: OpDecorate %[[#e7]] FPRoundingMode RTN

; CHECK-DAG: %[[#r1:]] = OpFAdd %[[#]] %[[#]]
; CHECK-DAG: %[[#r2:]] = OpFDiv %[[#]] %[[#]]
; CHECK-DAG: %[[#r3:]] = OpFSub %[[#]] %[[#]]
; CHECK-DAG: %[[#r4:]] = OpFMul %[[#]] %[[#]]
; CHECK-DAG: %[[#r5:]] = OpExtInst %[[#]] %[[#]] fma %[[#]] %[[#]] %[[#]]
; CHECK-DAG: %[[#r6:]] = OpFRem
; CHECK-DAG: %[[#r7:]] = OpExtInst %[[#]] %[[#]] sqrt

; CHECK-DAG: OpDecorate %[[#r1]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#r4]] FPRoundingMode RTN
; CHECK-DAG: OpDecorate %[[#r3]] FPRoundingMode RTP

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-NOEXT --implicit-check-not=FPRoundingMode
; TODO: remove the following check FPRoundingMode is placed correctly
; CHECK-NOEXT-COUNT-3: FPRoundingMode


@G_r1 = global float 0.0
@G_r2 = global float 0.0
@G_r3 = global float 0.0
@G_r4 = global float 0.0
@G_r5 = global float 0.0
@G_r6 = global float 0.0
@G_r7 = global float 0.0

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
  %r7 = tail call float @llvm.experimental.constrained.sqrt.f32(float %a, metadata !"round.downward", metadata !"fpexcept.strict")
  store float %r7, ptr @G_r7
  ret void
}

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fmuladd.f32(float, float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
declare float @llvm.experimental.constrained.frem.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
