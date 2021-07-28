; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu < %s | FileCheck %s

define void @strict_fp_reductions() {
; CHECK-LABEL: strict_fp_reductions
; CHECK-NEXT: Cost Model: Found an estimated cost of 17 for instruction: %fadd_v4f32 = call float @llvm.vector.reduce.fadd.v4f32(float 0.000000e+00, <4 x float> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 34 for instruction: %fadd_v8f32 = call float @llvm.vector.reduce.fadd.v8f32(float 0.000000e+00, <8 x float> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 7 for instruction: %fadd_v2f64 = call double @llvm.vector.reduce.fadd.v2f64(double 0.000000e+00, <2 x double> undef)
; CHECK-NEXT: Cost Model: Found an estimated cost of 14 for instruction: %fadd_v4f64 = call double @llvm.vector.reduce.fadd.v4f64(double 0.000000e+00, <4 x double> undef)
  %fadd_v4f32 = call float @llvm.vector.reduce.fadd.v4f32(float 0.0, <4 x float> undef)
  %fadd_v8f32 = call float @llvm.vector.reduce.fadd.v8f32(float 0.0, <8 x float> undef)
  %fadd_v2f64 = call double @llvm.vector.reduce.fadd.v2f64(double 0.0, <2 x double> undef)
  %fadd_v4f64 = call double @llvm.vector.reduce.fadd.v4f64(double 0.0, <4 x double> undef)

  ret void
}

declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)
declare float @llvm.vector.reduce.fadd.v8f32(float, <8 x float>)
declare double @llvm.vector.reduce.fadd.v2f64(double, <2 x double>)
declare double @llvm.vector.reduce.fadd.v4f64(double, <4 x double>)
