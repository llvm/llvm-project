; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <2 x float> @fma(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; CHECK-LABEL: fma:
; CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp1 = fmul contract <2 x float> %A, %B;
	%tmp2 = fadd contract <2 x float> %C, %tmp1;
	ret <2 x float> %tmp2
}

define <2 x float> @no_fma_1(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; CHECK-LABEL: no_fma_1:
; CHECK: fmul
; CHECK: fadd
	%tmp1 = fmul contract <2 x float> %A, %B;
	%tmp2 = fadd <2 x float> %C, %tmp1;
	ret <2 x float> %tmp2
}

define <2 x float> @no_fma_2(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; CHECK-LABEL: no_fma_2:
; CHECK: fmul
; CHECK: fadd
	%tmp1 = fmul <2 x float> %A, %B;
	%tmp2 = fadd contract <2 x float> %C, %tmp1;
	ret <2 x float> %tmp2
}

define <2 x float> @fma_sub(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; CHECK-LABEL: fma_sub:
; CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp1 = fmul contract <2 x float> %A, %B;
	%tmp2 = fsub contract <2 x float> %C, %tmp1;
	ret <2 x float> %tmp2
}

define <2 x float> @no_fma_sub_1(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; CHECK-LABEL: no_fma_sub_1:
; CHECK: fmul
; CHECK: fsub
	%tmp1 = fmul contract <2 x float> %A, %B;
	%tmp2 = fsub <2 x float> %C, %tmp1;
	ret <2 x float> %tmp2
}

define <2 x float> @no_fma_sub_2(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; CHECK-LABEL: no_fma_sub_2:
; CHECK: fmul
; CHECK: fsub
	%tmp1 = fmul <2 x float> %A, %B;
	%tmp2 = fsub contract <2 x float> %C, %tmp1;
	ret <2 x float> %tmp2
}
