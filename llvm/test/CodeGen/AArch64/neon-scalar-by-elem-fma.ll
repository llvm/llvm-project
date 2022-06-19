; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s

attributes #0 = { strictfp }

declare float @llvm.fma.f32(float, float, float)
declare double @llvm.fma.f64(double, double, double)
declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)

define float @test_fmla_ss4S(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss4S
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = call float @llvm.fma.f32(float %b, float %tmp1, float %a)
  ret float %tmp2
}

define float @test_fmla_ss4S_swap(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss4S_swap
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = call float @llvm.fma.f32(float %tmp1, float %a, float %a)
  ret float %tmp2
}

define float @test_fmla_ss2S(float %a, float %b, <2 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss2S
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
  %tmp1 = extractelement <2 x float> %v, i32 1
  %tmp2 = call float @llvm.fma.f32(float %b, float %tmp1, float %a)
  ret float %tmp2
}

define double @test_fmla_ddD(double %a, double %b, <1 x double> %v) {
  ; CHECK-LABEL: test_fmla_ddD
  ; CHECK: {{fmla d[0-9]+, d[0-9]+, v[0-9]+.d\[0]|fmadd d[0-9]+, d[0-9]+, d[0-9]+, d[0-9]+}}
  %tmp1 = extractelement <1 x double> %v, i32 0
  %tmp2 = call double @llvm.fma.f64(double %b, double %tmp1, double %a)
  ret double %tmp2
}

define double @test_fmla_dd2D(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmla_dd2D
  ; CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = call double @llvm.fma.f64(double %b, double %tmp1, double %a)
  ret double %tmp2
}

define double @test_fmla_dd2D_swap(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmla_dd2D_swap
  ; CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = call double @llvm.fma.f64(double %tmp1, double %b, double %a)
  ret double %tmp2
}

define float @test_fmls_ss4S(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss4S
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = fsub float -0.0, %tmp1
  %tmp3 = call float @llvm.fma.f32(float %tmp2, float %tmp1, float %a)
  ret float %tmp3
}

define float @test_fmls_ss4S_swap(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss4S_swap
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = fsub float -0.0, %tmp1
  %tmp3 = call float @llvm.fma.f32(float %tmp1, float %tmp2, float %a)
  ret float %tmp3
}


define float @test_fmls_ss2S(float %a, float %b, <2 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss2S
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
  %tmp1 = extractelement <2 x float> %v, i32 1
  %tmp2 = fsub float -0.0, %tmp1
  %tmp3 = call float @llvm.fma.f32(float %tmp2, float %tmp1, float %a)
  ret float %tmp3
}

define double @test_fmls_ddD(double %a, double %b, <1 x double> %v) {
  ; CHECK-LABEL: test_fmls_ddD
  ; CHECK: {{fmls d[0-9]+, d[0-9]+, v[0-9]+.d\[0]|fmsub d[0-9]+, d[0-9]+, d[0-9]+, d[0-9]+}}
  %tmp1 = extractelement <1 x double> %v, i32 0
  %tmp2 = fsub double -0.0, %tmp1
  %tmp3 = call double @llvm.fma.f64(double %tmp2, double %tmp1, double %a)
  ret double %tmp3
}

define double @test_fmls_dd2D(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmls_dd2D
  ; CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = fsub double -0.0, %tmp1
  %tmp3 = call double @llvm.fma.f64(double %tmp2, double %tmp1, double %a)
  ret double %tmp3
}

define double @test_fmls_dd2D_swap(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmls_dd2D_swap
  ; CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = fsub double -0.0, %tmp1
  %tmp3 = call double @llvm.fma.f64(double %tmp1, double %tmp2, double %a)
  ret double %tmp3
}

define float @test_fmla_ss4S_strict(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss4S_strict
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = call float @llvm.experimental.constrained.fma.f32(float %b, float %tmp1, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %tmp2
}

define float @test_fmla_ss4S_swap_strict(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss4S_swap_strict
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = call float @llvm.experimental.constrained.fma.f32(float %tmp1, float %a, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %tmp2
}

define float @test_fmla_ss2S_strict(float %a, float %b, <2 x float> %v) {
  ; CHECK-LABEL: test_fmla_ss2S_strict
  ; CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
  %tmp1 = extractelement <2 x float> %v, i32 1
  %tmp2 = call float @llvm.experimental.constrained.fma.f32(float %b, float %tmp1, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %tmp2
}

define double @test_fmla_ddD_strict(double %a, double %b, <1 x double> %v) {
  ; CHECK-LABEL: test_fmla_ddD_strict
  ; CHECK: {{fmla d[0-9]+, d[0-9]+, v[0-9]+.d\[0]|fmadd d[0-9]+, d[0-9]+, d[0-9]+, d[0-9]+}}
  %tmp1 = extractelement <1 x double> %v, i32 0
  %tmp2 = call double @llvm.experimental.constrained.fma.f64(double %b, double %tmp1, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %tmp2
}

define double @test_fmla_dd2D_strict(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmla_dd2D_strict
  ; CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = call double @llvm.experimental.constrained.fma.f64(double %b, double %tmp1, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %tmp2
}

define double @test_fmla_dd2D_swap_strict(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmla_dd2D_swap_strict
  ; CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = call double @llvm.experimental.constrained.fma.f64(double %tmp1, double %b, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %tmp2
}

define float @test_fmls_ss4S_strict(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss4S_strict
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = fneg float %tmp1
  %tmp3 = call float @llvm.experimental.constrained.fma.f32(float %tmp2, float %tmp1, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %tmp3
}

define float @test_fmls_ss4S_swap_strict(float %a, float %b, <4 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss4S_swap_strict
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
  %tmp1 = extractelement <4 x float> %v, i32 3
  %tmp2 = fneg float %tmp1
  %tmp3 = call float @llvm.experimental.constrained.fma.f32(float %tmp1, float %tmp2, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %tmp3
}

define float @test_fmls_ss2S_strict(float %a, float %b, <2 x float> %v) {
  ; CHECK-LABEL: test_fmls_ss2S_strict
  ; CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
  %tmp1 = extractelement <2 x float> %v, i32 1
  %tmp2 = fneg float %tmp1
  %tmp3 = call float @llvm.experimental.constrained.fma.f32(float %tmp2, float %tmp1, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %tmp3
}

define double @test_fmls_ddD_strict(double %a, double %b, <1 x double> %v) {
  ; CHECK-LABEL: test_fmls_ddD_strict
  ; CHECK: {{fmls d[0-9]+, d[0-9]+, v[0-9]+.d\[0]|fmsub d[0-9]+, d[0-9]+, d[0-9]+, d[0-9]+}}
  %tmp1 = extractelement <1 x double> %v, i32 0
  %tmp2 = fneg double %tmp1
  %tmp3 = call double @llvm.experimental.constrained.fma.f64(double %tmp2, double %tmp1, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %tmp3
}

define double @test_fmls_dd2D_strict(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmls_dd2D_strict
  ; CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = fneg double %tmp1
  %tmp3 = call double @llvm.experimental.constrained.fma.f64(double %tmp2, double %tmp1, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %tmp3
}

define double @test_fmls_dd2D_swap_strict(double %a, double %b, <2 x double> %v) {
  ; CHECK-LABEL: test_fmls_dd2D_swap_strict
  ; CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
  %tmp1 = extractelement <2 x double> %v, i32 1
  %tmp2 = fneg double %tmp1
  %tmp3 = call double @llvm.experimental.constrained.fma.f64(double %tmp1, double %tmp2, double %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret double %tmp3
}

