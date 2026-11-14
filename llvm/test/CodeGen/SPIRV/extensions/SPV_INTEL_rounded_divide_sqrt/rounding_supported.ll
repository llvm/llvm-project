; Verifies that FPRoundingMode decorations on OpFDiv and sqrt (OpenCL.std
; extended instruction) trigger the RoundedDivideSqrtINTEL capability and the
; SPV_INTEL_rounded_divide_sqrt extension.
; Some operand pairs are swapped operands to avoid MachineCSE dropping the
; 2nd instruction (see fdiv_cse.ll and
; https://github.com/llvm/llvm-project/issues/241441), such as the %d0 and %d1
; operands below.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt %s -o - | FileCheck %s
; RUN: llc -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_rounded_divide_sqrt %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability RoundedDivideSqrtINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_rounded_divide_sqrt"

; CHECK-DAG: %[[#HALF:]] = OpTypeFloat 16
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#DOUBLE:]] = OpTypeFloat 64
; CHECK-DAG: %[[#HALFV:]] = OpTypeVector %[[#HALF]] 2
; CHECK-DAG: %[[#FLOATV:]] = OpTypeVector %[[#FLOAT]] 4
; CHECK-DAG: %[[#DOUBLEV:]] = OpTypeVector %[[#DOUBLE]] 3

; CHECK-DAG: %[[#H_RTE:]] = OpFDiv %[[#HALF]]
; CHECK-DAG: %[[#F_RTZ:]] = OpFDiv %[[#FLOAT]]
; CHECK-DAG: %[[#D_RTP:]] = OpFDiv %[[#DOUBLE]]
; CHECK-DAG: %[[#D_RTN:]] = OpFDiv %[[#DOUBLE]]
; CHECK-DAG: OpDecorate %[[#H_RTE]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#F_RTZ]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#D_RTP]] FPRoundingMode RTP
; CHECK-DAG: OpDecorate %[[#D_RTN]] FPRoundingMode RTN
define spir_kernel void @test_fdiv_scalar(half %h0, half %h1, float %f0, float %f1, double %d0, double %d1) strictfp {
entry:
  %h_rte = call half @llvm.experimental.constrained.fdiv.f16(half %h0, half %h1, metadata !"round.tonearest", metadata !"fpexcept.strict")
  %f_rtz = call float @llvm.experimental.constrained.fdiv.f32(float %f0, float %f1, metadata !"round.towardzero", metadata !"fpexcept.strict")
  %d_rtp = call double @llvm.experimental.constrained.fdiv.f64(double %d0, double %d1, metadata !"round.upward", metadata !"fpexcept.strict")
  %d_rtn = call double @llvm.experimental.constrained.fdiv.f64(double %d1, double %d0, metadata !"round.downward", metadata !"fpexcept.strict")
  ret void
}

; CHECK-DAG: %[[#HV_RTE:]] = OpFDiv %[[#HALFV]]
; CHECK-DAG: %[[#FV_RTZ:]] = OpFDiv %[[#FLOATV]]
; CHECK-DAG: %[[#DV_RTP:]] = OpFDiv %[[#DOUBLEV]]
; CHECK-DAG: %[[#DV_RTN:]] = OpFDiv %[[#DOUBLEV]]
; CHECK-DAG: OpDecorate %[[#HV_RTE]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#FV_RTZ]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#DV_RTP]] FPRoundingMode RTP
; CHECK-DAG: OpDecorate %[[#DV_RTN]] FPRoundingMode RTN
define spir_kernel void @test_fdiv_vector(<2 x half> %h0, <2 x half> %h1, <4 x float> %f0, <4 x float> %f1, <3 x double> %d0, <3 x double> %d1) strictfp {
entry:
  %h_rte = call <2 x half> @llvm.experimental.constrained.fdiv.v2f16(<2 x half> %h0, <2 x half> %h1, metadata !"round.tonearest", metadata !"fpexcept.strict")
  %f_rtz = call <4 x float> @llvm.experimental.constrained.fdiv.v4f32(<4 x float> %f0, <4 x float> %f1, metadata !"round.towardzero", metadata !"fpexcept.strict")
  %d_rtp = call <3 x double> @llvm.experimental.constrained.fdiv.v3f64(<3 x double> %d0, <3 x double> %d1, metadata !"round.upward", metadata !"fpexcept.strict")
  %d_rtn = call <3 x double> @llvm.experimental.constrained.fdiv.v3f64(<3 x double> %d1, <3 x double> %d0, metadata !"round.downward", metadata !"fpexcept.strict")
  ret void
}

; CHECK-DAG: %[[#S_RTE:]] = OpExtInst %[[#HALF]] %[[#]] sqrt
; CHECK-DAG: %[[#S_RTZ:]] = OpExtInst %[[#FLOAT]] %[[#]] sqrt
; CHECK-DAG: %[[#S_RTP:]] = OpExtInst %[[#DOUBLE]] %[[#]] sqrt
; CHECK-DAG: %[[#S_RTN:]] = OpExtInst %[[#DOUBLE]] %[[#]] sqrt
; CHECK-DAG: OpDecorate %[[#S_RTE]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#S_RTZ]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#S_RTP]] FPRoundingMode RTP
; CHECK-DAG: OpDecorate %[[#S_RTN]] FPRoundingMode RTN
define spir_kernel void @test_sqrt_scalar(half %h, float %f, double %d0, double %d1) strictfp {
entry:
  %h_rte = call half @llvm.experimental.constrained.sqrt.f16(half %h, metadata !"round.tonearest", metadata !"fpexcept.strict")
  %f_rtz = call float @llvm.experimental.constrained.sqrt.f32(float %f, metadata !"round.towardzero", metadata !"fpexcept.strict")
  %d_rtp = call double @llvm.experimental.constrained.sqrt.f64(double %d0, metadata !"round.upward", metadata !"fpexcept.strict")
  %d_rtn = call double @llvm.experimental.constrained.sqrt.f64(double %d1, metadata !"round.downward", metadata !"fpexcept.strict")
  ret void
}

; CHECK-DAG: %[[#SV_RTE:]] = OpExtInst %[[#HALFV]] %[[#]] sqrt
; CHECK-DAG: %[[#SV_RTZ:]] = OpExtInst %[[#FLOATV]] %[[#]] sqrt
; CHECK-DAG: %[[#SV_RTP:]] = OpExtInst %[[#DOUBLEV]] %[[#]] sqrt
; CHECK-DAG: %[[#SV_RTN:]] = OpExtInst %[[#DOUBLEV]] %[[#]] sqrt
; CHECK-DAG: OpDecorate %[[#SV_RTE]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#SV_RTZ]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#SV_RTP]] FPRoundingMode RTP
; CHECK-DAG: OpDecorate %[[#SV_RTN]] FPRoundingMode RTN
define spir_kernel void @test_sqrt_vector(<2 x half> %h, <4 x float> %f, <3 x double> %d0, <3 x double> %d1) strictfp {
entry:
  %h_rte = call <2 x half> @llvm.experimental.constrained.sqrt.v2f16(<2 x half> %h, metadata !"round.tonearest", metadata !"fpexcept.strict")
  %f_rtz = call <4 x float> @llvm.experimental.constrained.sqrt.v4f32(<4 x float> %f, metadata !"round.towardzero", metadata !"fpexcept.strict")
  %d_rtp = call <3 x double> @llvm.experimental.constrained.sqrt.v3f64(<3 x double> %d0, metadata !"round.upward", metadata !"fpexcept.strict")
  %d_rtn = call <3 x double> @llvm.experimental.constrained.sqrt.v3f64(<3 x double> %d1, metadata !"round.downward", metadata !"fpexcept.strict")
  ret void
}

declare half @llvm.experimental.constrained.fdiv.f16(half, half, metadata, metadata)
declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)
declare <2 x half> @llvm.experimental.constrained.fdiv.v2f16(<2 x half>, <2 x half>, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.fdiv.v4f32(<4 x float>, <4 x float>, metadata, metadata)
declare <3 x double> @llvm.experimental.constrained.fdiv.v3f64(<3 x double>, <3 x double>, metadata, metadata)

declare half @llvm.experimental.constrained.sqrt.f16(half, metadata, metadata)
declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)
declare <2 x half> @llvm.experimental.constrained.sqrt.v2f16(<2 x half>, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.sqrt.v4f32(<4 x float>, metadata, metadata)
declare <3 x double> @llvm.experimental.constrained.sqrt.v3f64(<3 x double>, metadata, metadata)
