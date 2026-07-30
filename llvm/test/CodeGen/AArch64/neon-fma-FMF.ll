; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s
; RUN: llc < %s -O3 -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <2 x float> @fma_1(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; CHECK-LABEL: fma_1:
; CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp1 = fmul contract <2 x float> %A, %B;
	%tmp2 = fadd contract <2 x float> %C, %tmp1;
	ret <2 x float> %tmp2
}

; This case will fold as it was only available through unsafe before, now available from
; the contract on the fadd
define <2 x float> @fma_2(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; CHECK-LABEL: fma_2:
; CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp1 = fmul <2 x float> %A, %B;
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

define <2 x float> @fma_sub_1(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; CHECK-LABEL: fma_sub_1:
; CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp1 = fmul contract <2 x float> %A, %B;
	%tmp2 = fsub contract <2 x float> %C, %tmp1;
	ret <2 x float> %tmp2
}

; This case will fold as it was only available through unsafe before, now available from
; the contract on the fsub
define <2 x float> @fma_sub_2(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; CHECK-LABEL: fma_sub_2:
; CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp1 = fmul <2 x float> %A, %B;
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

; Regression test: contract FMF allows folding (A * 0 + B) to FMA(A, 0, B), but
; reassoc FMF must not allow further folding to just (B) without additional
; FMFs (ninf, nnan)
define float @fma_zero(float %A, float %B) {
; CHECK-LABEL: fma_zero:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movi d2, #0000000000000000
; CHECK-NEXT:    fmadd s0, s0, s2, s1
; CHECK-NEXT:    ret
	%tmp1 = fmul contract reassoc float %A, 0.0e+0;
	%tmp2 = fadd contract reassoc float %B, %tmp1;
	ret float %tmp2
}
