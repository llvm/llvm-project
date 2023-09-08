; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s

attributes #0 = { strictfp }

declare float @llvm.fma.f32(float, float, float)
declare double @llvm.fma.f64(double, double, double)
declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)

define float @test_fmla_ss4S_0(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss4S_0
  ; CHECK: fmadd s0, s1, s2, s0
  %tmp1 = extractelement <4 x float> %v, i32 0
  %tmp2 = call float @llvm.fma.f32(float %b, float %tmp1, float %a)
  ret float %tmp2
}

define float @test_fmla_ss4S_0_swap(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss4S_0_swap
  ; CHECK: fmadd s0, s2, s1, s0
  %tmp1 = extractelement <4 x float> %v, i32 0
  %tmp2 = call float @llvm.fma.f32(float %tmp1, float %b, float %a)
  ret float %tmp2
}

define float @test_fmla_ss4S_3(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss4S_3
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = call float @llvm.fma.f32(float %b, float %tmp1, float %a)
  ret float %tmp2
}

define float @test_fmla_ss4S_3_swap(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss4S_3_swap
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = call float @llvm.fma.f32(float %tmp1, float %a, float %a)
  ret float %tmp2
}

define float @test_fmla_ss2S_0(float %a, float %b, <2 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss2S_0
  ; CHECK: fmadd s0, s1, s2, s0
  %tmp1 = extractelement <2 x float> %v, i32 0
  %tmp2 = call float @llvm.fma.f32(float %b, float %tmp1, float %a)
  ret float %tmp2
}

define float @test_fmla_ss2S_0_swap(float %a, float %b, <2 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss2S_0_swap
  ; CHECK: fmadd s0, s2, s1, s0
  %tmp1 = extractelement <2 x float> %v, i32 0
  %tmp2 = call float @llvm.fma.f32(float %tmp1, float %b, float %a)
  ret float %tmp2
}

define float @test_fmla_ss2S_1(float %a, float %b, <2 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss2S_1
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
  %tmp1 = extractelement <2 x float> %v, i32 1
  %tmp2 = call float @llvm.fma.f32(float %b, float %tmp1, float %a)
  ret float %tmp2
}

define double @test_fmla_ddD_0(double %a, double %b, <1 x double> %v) {
  ; CHECK-LABEL: test_fmla_ddD_0
  ; CHECK: fmadd d0, d1, d2, d0
  %tmp1 = extractelement <1 x double> %v, i32 0
  %tmp2 = call double @llvm.fma.f64(double %b, double %tmp1, double %a)
  ret double %tmp2
}

define double @test_fmla_ddD_0_swap(double %a, double %b, <1 x double> %v) {
  ; CHECK-LABEL: test_fmla_ddD_0_swap
  ; CHECK: fmadd d0, d2, d1, d0
  %tmp1 = extractelement <1 x double> %v, i32 0
  %tmp2 = call double @llvm.fma.f64(double %tmp1, double %b, double %a)
  ret double %tmp2
}

define double @test_fmla_dd2D_0(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmla_dd2D_0
  ; CHECK: fmadd d0, d1, d2, d0
  %tmp1 = extractelement <2 x double> %v, i32 0
  %tmp2 = call double @llvm.fma.f64(double %b, double %tmp1, double %a)
  ret double %tmp2
}

define double @test_fmla_dd2D_0_swap(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmla_dd2D_0_swap
  ; CHECK: fmadd d0, d2, d1, d0
  %tmp1 = extractelement <2 x double> %v, i32 0
  %tmp2 = call double @llvm.fma.f64(double %tmp1, double %b, double %a)
  ret double %tmp2
}

define double @test_fmla_dd2D_1(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmla_dd2D_1
  ; CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = call double @llvm.fma.f64(double %b, double %tmp1, double %a)
  ret double %tmp2
}

define double @test_fmla_dd2D_1_swap(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmla_dd2D_1_swap
  ; CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = call double @llvm.fma.f64(double %tmp1, double %b, double %a)
  ret double %tmp2
}

define float @test_fmls_ss4S_0(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss4S_0
  ; CHECK: fmsub s0, s2, s1, s0
entry:
  %fneg = fneg float %b
  %extract = extractelement <4 x float> %v, i64 0
  %0 = tail call float @llvm.fma.f32(float %fneg, float %extract, float %a)
  ret float %0
}

define float @test_fmls_ss4S_0_swap(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss4S_0_swap
  ; CHECK: fmsub s0, s2, s1, s0
entry:
  %fneg = fneg float %b
  %extract = extractelement <4 x float> %v, i64 0
  %0 = tail call float @llvm.fma.f32(float %extract, float %fneg, float %a)
  ret float %0
}

define float @test_fmls_ss4S_3(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss4S_3
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = fsub float -0.0, %tmp1
  %tmp3 = call float @llvm.fma.f32(float %tmp2, float %tmp1, float %a)
  ret float %tmp3
}

define float @test_fmls_ss4S_3_swap(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss4S_3_swap
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = fsub float -0.0, %tmp1
  %tmp3 = call float @llvm.fma.f32(float %tmp1, float %tmp2, float %a)
  ret float %tmp3
}


define float @test_fmls_ss2S_0(float %a, float %b, <2 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss2S_0
  ; CHECK: fmsub s0, s2, s1, s0
entry:
  %fneg = fneg float %b
  %extract = extractelement <2 x float> %v, i64 0
  %0 = tail call float @llvm.fma.f32(float %fneg, float %extract, float %a)
  ret float %0
}

define float @test_fmls_ss2S_0_swap(float %a, float %b, <2 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss2S_0_swap
  ; CHECK: fmsub s0, s2, s1, s0
entry:
  %fneg = fneg float %b
  %extract = extractelement <2 x float> %v, i64 0
  %0 = tail call float @llvm.fma.f32(float %extract, float %fneg, float %a)
  ret float %0
}

define float @test_fmls_ss2S_1(float %a, float %b, <2 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss2S_1
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
  %tmp1 = extractelement <2 x float> %v, i32 1
  %tmp2 = fsub float -0.0, %tmp1
  %tmp3 = call float @llvm.fma.f32(float %tmp2, float %tmp1, float %a)
  ret float %tmp3
}

define double @test_fmls_ddD_0(double %a, double %b, <1 x double> %v) {
  ; CHECK-LABEL: test_fmls_ddD_0
  ; CHECK: fmsub d0, d1, d2, d0
entry:
  %fneg = fneg double %b
  %extract = extractelement <1 x double> %v, i64 0
  %0 = tail call double @llvm.fma.f64(double %fneg, double %extract, double %a)
  ret double %0
}

define double @test_fmls_ddD_0_swap(double %a, double %b, <1 x double> %v) {
  ; CHECK-LABEL: test_fmls_ddD_0_swap
  ; CHECK: fmsub d0, d2, d1, d0
entry:
  %fneg = fneg double %b
  %extract = extractelement <1 x double> %v, i64 0
  %0 = tail call double @llvm.fma.f64(double %extract, double %fneg, double %a)
  ret double %0
}

define double @test_fmls_dd2D_0(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmls_dd2D_0
  ; CHECK: fmsub d0, d2, d1, d0
entry:
  %fneg = fneg double %b
  %extract = extractelement <2 x double> %v, i64 0
  %0 = tail call double @llvm.fma.f64(double %fneg, double %extract, double %a)
  ret double %0
}

define double @test_fmls_dd2D_0_swap(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmls_dd2D_0_swap
  ; CHECK: fmsub d0, d2, d1, d0
entry:
  %fneg = fneg double %b
  %extract = extractelement <2 x double> %v, i64 0
  %0 = tail call double @llvm.fma.f64(double %extract, double %fneg, double %a)
  ret double %0
}

define double @test_fmls_dd2D_1(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmls_dd2D_1
  ; CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = fsub double -0.0, %tmp1
  %tmp3 = call double @llvm.fma.f64(double %tmp2, double %tmp1, double %a)
  ret double %tmp3
}

define double @test_fmls_dd2D_1_swap(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmls_dd2D_1_swap
  ; CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = fsub double -0.0, %tmp1
  %tmp3 = call double @llvm.fma.f64(double %tmp1, double %tmp2, double %a)
  ret double %tmp3
}

define float @test_fmla_ss4S_0_strict(float %a, float %b, <4 x float> %v) #0 {
  ; CHECK-LABEL: test_fmla_ss4S_0_strict
  ; CHECK: fmadd s0, s1, s2, s0
  %tmp1 = extractelement <4 x float> %v, i32 0
  %tmp2 = call float @llvm.experimental.constrained.fma.f32(float %b, float %tmp1, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %tmp2
}

define float @test_fmla_ss4S_0_swap_strict(float %a, float %b, <4 x float> %v) #0 {
  ; CHECK-LABEL: test_fmla_ss4S_0_swap_strict
  ; CHECK: fmadd s0, s2, s1, s0
  %tmp1 = extractelement <4 x float> %v, i32 0
  %tmp2 = call float @llvm.experimental.constrained.fma.f32(float %tmp1, float %b, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %tmp2
}

define float @test_fmla_ss4S_3_strict(float %a, float %b, <4 x float> %v) #0 {
  ; CHECK-LABEL: test_fmla_ss4S_3_strict
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = call float @llvm.experimental.constrained.fma.f32(float %b, float %tmp1, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %tmp2
}

define float @test_fmla_ss4S_3_swap_strict(float %a, float %b, <4 x float> %v) #0 {
  ; CHECK-LABEL: test_fmla_ss4S_3_swap_strict
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = call float @llvm.experimental.constrained.fma.f32(float %tmp1, float %a, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %tmp2
}

define float @test_fmla_ss2S_0_strict(float %a, float %b, <2 x float> %v) #0 {
  ; CHECK-LABEL: test_fmla_ss2S_0_strict
  ; CHECK: fmadd s0, s1, s2, s0
  %tmp1 = extractelement <2 x float> %v, i32 0
  %tmp2 = call float @llvm.experimental.constrained.fma.f32(float %b, float %tmp1, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %tmp2
}

define float @test_fmla_ss2S_0_swap_strict(float %a, float %b, <2 x float> %v) #0 {
  ; CHECK-LABEL: test_fmla_ss2S_0_swap_strict
  ; CHECK: fmadd s0, s2, s1, s0
  %tmp1 = extractelement <2 x float> %v, i32 0
  %tmp2 = call float @llvm.experimental.constrained.fma.f32(float %tmp1, float %b, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %tmp2
}

define float @test_fmla_ss2S_1_strict(float %a, float %b, <2 x float> %v) #0 {
  ; CHECK-LABEL: test_fmla_ss2S_1_strict
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
  %tmp1 = extractelement <2 x float> %v, i32 1
  %tmp2 = call float @llvm.experimental.constrained.fma.f32(float %b, float %tmp1, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %tmp2
}

define double @test_fmla_ddD_0_strict(double %a, double %b, <1 x double> %v) #0 {
  ; CHECK-LABEL: test_fmla_ddD_0_strict
  ; CHECK: fmadd d0, d1, d2, d0
  %tmp1 = extractelement <1 x double> %v, i32 0
  %tmp2 = call double @llvm.experimental.constrained.fma.f64(double %b, double %tmp1, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %tmp2
}

define double @test_fmla_ddD_0_swap_strict(double %a, double %b, <1 x double> %v) #0 {
  ; CHECK-LABEL: test_fmla_ddD_0_swap_strict
  ; CHECK: fmadd d0, d2, d1, d0
  %tmp1 = extractelement <1 x double> %v, i32 0
  %tmp2 = call double @llvm.experimental.constrained.fma.f64(double %tmp1, double %b, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %tmp2
}

define double @test_fmla_dd2D_0_strict(double %a, double %b, <2 x double> %v) #0 {
  ; CHECK-LABEL: test_fmla_dd2D_0_strict
  ; CHECK: fmadd d0, d1, d2, d0
  %tmp1 = extractelement <2 x double> %v, i32 0
  %tmp2 = call double @llvm.experimental.constrained.fma.f64(double %b, double %tmp1, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %tmp2
}

define double @test_fmla_dd2D_0_swap_strict(double %a, double %b, <2 x double> %v) #0 {
  ; CHECK-LABEL: test_fmla_dd2D_0_swap_strict
  ; CHECK: fmadd d0, d2, d1, d0
  %tmp1 = extractelement <2 x double> %v, i32 0
  %tmp2 = call double @llvm.experimental.constrained.fma.f64(double %tmp1, double %b, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %tmp2
}

define double @test_fmla_dd2D_1_strict(double %a, double %b, <2 x double> %v) #0 {
  ; CHECK-LABEL: test_fmla_dd2D_1_strict
  ; CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = call double @llvm.experimental.constrained.fma.f64(double %b, double %tmp1, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %tmp2
}

define double @test_fmla_dd2D_1_swap_strict(double %a, double %b, <2 x double> %v) #0 {
  ; CHECK-LABEL: test_fmla_dd2D_1_swap_strict
  ; CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = call double @llvm.experimental.constrained.fma.f64(double %tmp1, double %b, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %tmp2
}

define float @test_fmls_ss4S_0_strict(float %a, float %b, <4 x float> %v) #0 {
  ; CHECK-LABEL: test_fmls_ss4S_0_strict
  ; CHECK: fmsub s0, s2, s1, s0
entry:
  %fneg = fneg float %b
  %extract = extractelement <4 x float> %v, i64 0
  %0 = tail call float @llvm.experimental.constrained.fma.f32(float %fneg, float %extract, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %0
}

define float @test_fmls_ss4S_0_swap_strict(float %a, float %b, <4 x float> %v) #0 {
  ; CHECK-LABEL: test_fmls_ss4S_0_swap_strict
  ; CHECK: fmsub s0, s2, s1, s0
entry:
  %fneg = fneg float %b
  %extract = extractelement <4 x float> %v, i64 0
  %0 = tail call float @llvm.experimental.constrained.fma.f32(float %extract, float %fneg, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %0
}

define float @test_fmls_ss4S_3_strict(float %a, float %b, <4 x float> %v) #0 {
  ; CHECK-LABEL: test_fmls_ss4S_3_strict
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = fneg float %tmp1
  %tmp3 = call float @llvm.experimental.constrained.fma.f32(float %tmp2, float %tmp1, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %tmp3
}

define float @test_fmls_ss4S_3_swap_strict(float %a, float %b, <4 x float> %v) #0 {
  ; CHECK-LABEL: test_fmls_ss4S_3_swap_strict
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = fneg float %tmp1
  %tmp3 = call float @llvm.experimental.constrained.fma.f32(float %tmp1, float %tmp2, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %tmp3
}

define float @test_fmls_ss2S_0_strict(float %a, float %b, <2 x float> %v) #0 {
  ; CHECK-LABEL: test_fmls_ss2S_0_strict
  ; CHECK: fmsub s0, s2, s1, s0
entry:
  %fneg = fneg float %b
  %extract = extractelement <2 x float> %v, i64 0
  %0 = tail call float @llvm.experimental.constrained.fma.f32(float %fneg, float %extract, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %0
}

define float @test_fmls_ss2S_0_swap_strict(float %a, float %b, <2 x float> %v) #0 {
  ; CHECK-LABEL: test_fmls_ss2S_0_swap_strict
  ; CHECK: fmsub s0, s2, s1, s0
entry:
  %fneg = fneg float %b
  %extract = extractelement <2 x float> %v, i64 0
  %0 = tail call float @llvm.experimental.constrained.fma.f32(float %extract, float %fneg, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %0
}

define float @test_fmls_ss2S_1_strict(float %a, float %b, <2 x float> %v) #0 {
  ; CHECK-LABEL: test_fmls_ss2S_1_strict
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
  %tmp1 = extractelement <2 x float> %v, i32 1
  %tmp2 = fneg float %tmp1
  %tmp3 = call float @llvm.experimental.constrained.fma.f32(float %tmp2, float %tmp1, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %tmp3
}

define double @test_fmls_ddD_0_strict(double %a, double %b, <1 x double> %v) #0 {
  ; CHECK-LABEL: test_fmls_ddD_0_strict
  ; CHECK: fmsub d0, d2, d1, d0
entry:
  %fneg = fneg double %b
  %extract = extractelement <1 x double> %v, i64 0
  %0 = tail call double @llvm.experimental.constrained.fma.f64(double %fneg, double %extract, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %0
}

define double @test_fmls_ddD_0_swap_strict(double %a, double %b, <1 x double> %v) #0 {
  ; CHECK-LABEL: test_fmls_ddD_0_swap_strict
  ; CHECK: fmsub d0, d2, d1, d0
entry:
  %fneg = fneg double %b
  %extract = extractelement <1 x double> %v, i64 0
  %0 = tail call double @llvm.experimental.constrained.fma.f64(double %extract, double %fneg, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %0
}

define double @test_fmls_dd2D_0_strict(double %a, double %b, <2 x double> %v) #0 {
  ; CHECK-LABEL: test_fmls_dd2D_0_strict
  ; CHECK: fmsub d0, d2, d1, d0
entry:
  %fneg = fneg double %b
  %extract = extractelement <2 x double> %v, i64 0
  %0 = tail call double @llvm.experimental.constrained.fma.f64(double %fneg, double %extract, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %0
}

define double @test_fmls_dd2D_0_swap_strict(double %a, double %b, <2 x double> %v) #0 {
  ; CHECK-LABEL: test_fmls_dd2D_0_swap_strict
  ; CHECK: fmsub d0, d2, d1, d0
entry:
  %fneg = fneg double %b
  %extract = extractelement <2 x double> %v, i64 0
  %0 = tail call double @llvm.experimental.constrained.fma.f64(double %extract, double %fneg, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %0
}

define double @test_fmls_dd2D_1_strict(double %a, double %b, <2 x double> %v) #0 {
  ; CHECK-LABEL: test_fmls_dd2D_1_strict
  ; CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = fneg double %tmp1
  %tmp3 = call double @llvm.experimental.constrained.fma.f64(double %tmp2, double %tmp1, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %tmp3
}

define double @test_fmls_dd2D_1_swap_strict(double %a, double %b, <2 x double> %v) #0 {
  ; CHECK-LABEL: test_fmls_dd2D_1_swap_strict
  ; CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = fneg double %tmp1
  %tmp3 = call double @llvm.experimental.constrained.fma.f64(double %tmp1, double %tmp2, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %tmp3
}

