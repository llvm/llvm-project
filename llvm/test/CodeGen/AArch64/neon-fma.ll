; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s

define <2 x float> @fmla2xfloat(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
;CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp1 = fmul <2 x float> %A, %B;
	%tmp2 = fadd <2 x float> %C, %tmp1;
	ret <2 x float> %tmp2
}

define <4 x float> @fmla4xfloat(<4 x float> %A, <4 x float> %B, <4 x float> %C) {
;CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp1 = fmul <4 x float> %A, %B;
	%tmp2 = fadd <4 x float> %C, %tmp1;
	ret <4 x float> %tmp2
}

define <2 x double> @fmla2xdouble(<2 x double> %A, <2 x double> %B, <2 x double> %C) {
;CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp1 = fmul <2 x double> %A, %B;
	%tmp2 = fadd <2 x double> %C, %tmp1;
	ret <2 x double> %tmp2
}


define <2 x float> @fmls2xfloat(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
;CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp1 = fmul <2 x float> %A, %B;
	%tmp2 = fsub <2 x float> %C, %tmp1;
	ret <2 x float> %tmp2
}

define <4 x float> @fmls4xfloat(<4 x float> %A, <4 x float> %B, <4 x float> %C) {
;CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp1 = fmul <4 x float> %A, %B;
	%tmp2 = fsub <4 x float> %C, %tmp1;
	ret <4 x float> %tmp2
}

define <2 x double> @fmls2xdouble(<2 x double> %A, <2 x double> %B, <2 x double> %C) {
;CHECK: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp1 = fmul <2 x double> %A, %B;
	%tmp2 = fsub <2 x double> %C, %tmp1;
	ret <2 x double> %tmp2
}


; Another set of tests for when the intrinsic is used.

declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>)

define <2 x float> @fmla2xfloat_fused(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
;CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
        %val = call <2 x float> @llvm.fma.v2f32(<2 x float> %A, <2 x float> %B, <2 x float> %C)
	ret <2 x float> %val
}

define <4 x float> @fmla4xfloat_fused(<4 x float> %A, <4 x float> %B, <4 x float> %C) {
;CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
        %val = call <4 x float> @llvm.fma.v4f32(<4 x float> %A, <4 x float> %B, <4 x float> %C)
	ret <4 x float> %val
}

define <2 x double> @fmla2xdouble_fused(<2 x double> %A, <2 x double> %B, <2 x double> %C) {
;CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
        %val = call <2 x double> @llvm.fma.v2f64(<2 x double> %A, <2 x double> %B, <2 x double> %C)
	ret <2 x double> %val
}

define <2 x float> @fmls2xfloat_fused(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
;CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
        %negA = fsub <2 x float> <float -0.0, float -0.0>, %A
        %val = call <2 x float> @llvm.fma.v2f32(<2 x float> %negA, <2 x float> %B, <2 x float> %C)
	ret <2 x float> %val
}

define <4 x float> @fmls4xfloat_fused(<4 x float> %A, <4 x float> %B, <4 x float> %C) {
;CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
        %negA = fsub <4 x float> <float -0.0, float -0.0, float -0.0, float -0.0>, %A
        %val = call <4 x float> @llvm.fma.v4f32(<4 x float> %negA, <4 x float> %B, <4 x float> %C)
	ret <4 x float> %val
}

define <2 x double> @fmls2xdouble_fused(<2 x double> %A, <2 x double> %B, <2 x double> %C) {
;CHECK: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
        %negA = fsub <2 x double> <double -0.0, double -0.0>, %A
        %val = call <2 x double> @llvm.fma.v2f64(<2 x double> %negA, <2 x double> %B, <2 x double> %C)
	ret <2 x double> %val
}

declare <2 x float> @llvm.fmuladd.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>)

define <2 x float> @fmuladd2xfloat(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
;CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
        %val = call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %A, <2 x float> %B, <2 x float> %C)
	ret <2 x float> %val
}

define <4 x float> @fmuladd4xfloat_fused(<4 x float> %A, <4 x float> %B, <4 x float> %C) {
;CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
        %val = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %A, <4 x float> %B, <4 x float> %C)
	ret <4 x float> %val
}

define <2 x double> @fmuladd2xdouble_fused(<2 x double> %A, <2 x double> %B, <2 x double> %C) {
;CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
        %val = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %A, <2 x double> %B, <2 x double> %C)
	ret <2 x double> %val
}


; Another set of tests that check for multiply single use

define <2 x float> @fmla2xfloati_su(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
;CHECK-NOT: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  %tmp1 = fmul <2 x float> %A, %B;
  %tmp2 = fadd <2 x float> %C, %tmp1;
  %tmp3 = fadd <2 x float> %tmp2, %tmp1;
  ret <2 x float> %tmp3
}

define <2 x double> @fmls2xdouble_su(<2 x double> %A, <2 x double> %B, <2 x double> %C) {
;CHECK-NOT: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
        %tmp1 = fmul <2 x double> %A, %B;
        %tmp2 = fsub <2 x double> %C, %tmp1;
        %tmp3 = fsub <2 x double> %tmp2, %tmp1;
        ret <2 x double> %tmp3
}

