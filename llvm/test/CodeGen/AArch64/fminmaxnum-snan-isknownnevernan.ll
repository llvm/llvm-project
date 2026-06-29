; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

;; SelectionDAG::isKnownNeverNaN answers "is this value never a signaling NaN?"
;; for FMINNUM/FMAXNUM/FMINIMUMNUM/FMAXIMUMNUM. These do not quiet a signaling
;; NaN: when an operand is a signaling NaN it may be returned unchanged, so the
;; result is known-never-signaling only if *both* operands are. The old code
;; used OR (it sufficed for one operand to be known never-NaN), which is correct
;; for the plain "never any NaN" query but unsound for the signaling query.
;;
;; This is observable through the bf16 narrowing in FP_TO_BF16 expansion, which
;; inserts an FCANONICALIZE (quieting) step unless the source is known never to
;; be a signaling NaN. On AArch64 that quieting lowers to setting the f32 quiet
;; bit (orr #0x400000) under an unordered check (fcmp + csel vs).

;; One operand is canonicalized (known never-sNaN), the other is unknown. The
;; minnum result may be the unknown operand, so a signaling NaN can survive and
;; the narrowing must quiet it. With the buggy OR-logic the source was wrongly
;; proven never-sNaN and the quieting was dropped.
; CHECK-LABEL: mixed:
; CHECK:       fminnm s0, s0, s1
; CHECK:       orr {{w[0-9]+}}, {{w[0-9]+}}, #0x400000
; CHECK:       csel {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, vs
define bfloat @mixed(float %a, float %b) {
  %na = call float @llvm.canonicalize.f32(float %a)
  %m  = call float @llvm.minnum.f32(float %na, float %b)
  %r  = fptrunc float %m to bfloat
  ret bfloat %r
}

;; Both operands are canonicalized, so the minnum result really is never a
;; signaling NaN and the narrowing may skip quieting. This guards against the
;; fix becoming over-broad (the AND must still fold to true here).
; CHECK-LABEL: both:
; CHECK-NOT:   #0x400000
define bfloat @both(float %a, float %b) {
  %na = call float @llvm.canonicalize.f32(float %a)
  %nb = call float @llvm.canonicalize.f32(float %b)
  %m  = call float @llvm.minnum.f32(float %na, float %nb)
  %r  = fptrunc float %m to bfloat
  ret bfloat %r
}

declare float @llvm.canonicalize.f32(float)
declare float @llvm.minnum.f32(float, float)
