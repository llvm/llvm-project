; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_100 -fp-contract=fast | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_100 | FileCheck %s --check-prefix=DEFAULT
; RUN: %if ptxas-12.8 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_100 -fp-contract=fast | %ptxas-verify -arch=sm_100 %}
; RUN: %if ptxas-12.8 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_100 | %ptxas-verify -arch=sm_100 %}

target triple = "nvptx64-unknown-cuda"

;; Make sure we are generating proper instruction sequences for fused ops
;; If fusion is allowed, we try to form fma.rn at the PTX level, and emit
;; add.f32 otherwise.  Without an explicit rounding mode on add.f32, ptxas
;; is free to fuse with a multiply if it is able.  If fusion is not allowed,
;; we do not form fma.rn at the PTX level and explicitly generate add.rn
;; for all adds to prevent ptxas from fusion the ops.

;; FAST-LABEL: @t0
;; DEFAULT-LABEL: @t0
define float @t0(float %a, float %b, float %c) {
;; FAST: fma.rn.f32
;; DEFAULT: mul.rn.f32
;; DEFAULT: add.rn.f32
  %v0 = fmul float %a, %b
  %v1 = fadd float %v0, %c
  ret float %v1
}

;; FAST-LABEL: @t1
;; DEFAULT-LABEL: @t1
define float @t1(float %a, float %b) {
;; We cannot form an fma here, but make sure we explicitly emit add.rn.f32
;; to prevent ptxas from fusing this with anything else.
;; FAST: add.f32
;; DEFAULT: add.rn.f32
  %v1 = fadd float %a, %b
  ret float %v1
}

;; FAST-LABEL: @t0_v2
;; DEFAULT-LABEL: @t0_v2
define <2 x float> @t0_v2(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
;; FAST: fma.rn.f32x2
;; DEFAULT: mul.rn.f32x2
;; DEFAULT: add.rn.f32x2
  %v0 = fmul <2 x float> %a, %b
  %v1 = fadd <2 x float> %v0, %c
  ret <2 x float> %v1
}

;; FAST-LABEL: @t1_v2
;; DEFAULT-LABEL: @t1_v2
define <2 x float> @t1_v2(<2 x float> %a, <2 x float> %b) {
;; We cannot form an fma here, but make sure we explicitly emit add.rn.f32
;; to prevent ptxas from fusing this with anything else.
;; FAST: add.f32
;; DEFAULT: add.rn.f32
  %v1 = fadd <2 x float> %a, %b
  ret <2 x float> %v1
}
