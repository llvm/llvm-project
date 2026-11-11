; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b %s -o - | FileCheck %s
;
; Tests that llvm.minnum/maxnum / llvm.vector.reduce.fmin/fmax correctly return
; NaN when both (all) inputs are qNaN, per llvm.minnum/maxnum semantics.
;
; The backend replaces NaN with the neutral value (+/-Inf) before the hardware
; min/max, which ignores single-operand NaN correctly but turns all-NaN inputs
; into the neutral value.  A post-operation fixup restores NaN in those lanes.
;
; For VECREDUCE_FMIN/FMAX: the predicate of non-NaN elements is collapsed to a
; scalar integer; if it is zero (all NaN) the scalar result is overridden with
; a qNaN constant.
;
; For FMINNUM/FMAXNUM: the BothNaN predicate (AND of per-lane IsNaN masks)
; selects the qNaN constant back into the result vector.
;
; With nnan the fixup is dead and must not be emitted.

; CHECK-LABEL: reduce_fmin_v32f32:
; CHECK:       vmin
; Fixup: ordered-predicate -> scalar integer -> if zero select qNaN.
; 0x7FC00000 = 2143289344 is the f32 quiet NaN.
; CHECK:       cmp.eq({{r[0-9]+}},#0)
; CHECK:       ##2143289344
; CHECK:       .Lfunc_end{{[0-9]+}}:
define float @reduce_fmin_v32f32(<32 x float> %x) {
  %r = call float @llvm.vector.reduce.fmin.v32f32(<32 x float> %x)
  ret float %r
}

; With nnan the fixup is unnecessary.
; CHECK-LABEL: reduce_fmin_v32f32_nnan:
; CHECK:       vmin
; CHECK-NOT:   ##2143289344
; CHECK:       .Lfunc_end{{[0-9]+}}:
define float @reduce_fmin_v32f32_nnan(<32 x float> %x) {
  %r = call nnan float @llvm.vector.reduce.fmin.v32f32(<32 x float> %x)
  ret float %r
}

; CHECK-LABEL: fminnum_v32f32:
; CHECK:       ##2143289344
; Fixup: BothNaN = IsNaN(a) AND IsNaN(b).
; CHECK:       and(
; CHECK:       vmin
; CHECK:       vmux
; CHECK:       .Lfunc_end{{[0-9]+}}:
define <32 x float> @fminnum_v32f32(<32 x float> %a, <32 x float> %b) {
  %r = call <32 x float> @llvm.minnum.v32f32(<32 x float> %a, <32 x float> %b)
  ret <32 x float> %r
}

; With nnan the fixup is unnecessary.
; CHECK-LABEL: fminnum_v32f32_nnan:
; CHECK:       vmin
; CHECK-NOT:   ##2143289344
; CHECK:       .Lfunc_end{{[0-9]+}}:
define <32 x float> @fminnum_v32f32_nnan(<32 x float> %a, <32 x float> %b) {
  %r = call nnan <32 x float> @llvm.minnum.v32f32(<32 x float> %a, <32 x float> %b)
  ret <32 x float> %r
}

; CHECK-LABEL: reduce_fmax_v32f32:
; CHECK:       vmax
; Fixup: ordered-predicate -> scalar integer -> if zero select qNaN.
; CHECK:       cmp.eq({{r[0-9]+}},#0)
; CHECK:       ##2143289344
; CHECK:       .Lfunc_end{{[0-9]+}}:
define float @reduce_fmax_v32f32(<32 x float> %x) {
  %r = call float @llvm.vector.reduce.fmax.v32f32(<32 x float> %x)
  ret float %r
}

; With nnan the fixup is unnecessary.
; CHECK-LABEL: reduce_fmax_v32f32_nnan:
; CHECK:       vmax
; CHECK-NOT:   ##2143289344
; CHECK:       .Lfunc_end{{[0-9]+}}:
define float @reduce_fmax_v32f32_nnan(<32 x float> %x) {
  %r = call nnan float @llvm.vector.reduce.fmax.v32f32(<32 x float> %x)
  ret float %r
}

; CHECK-LABEL: fmaxnum_v32f32:
; CHECK:       ##2143289344
; Fixup: BothNaN = IsNaN(a) AND IsNaN(b).
; CHECK:       and(
; CHECK:       vmax
; CHECK:       vmux
; CHECK:       .Lfunc_end{{[0-9]+}}:
define <32 x float> @fmaxnum_v32f32(<32 x float> %a, <32 x float> %b) {
  %r = call <32 x float> @llvm.maxnum.v32f32(<32 x float> %a, <32 x float> %b)
  ret <32 x float> %r
}

; With nnan the fixup is unnecessary.
; CHECK-LABEL: fmaxnum_v32f32_nnan:
; CHECK:       vmax
; CHECK-NOT:   ##2143289344
; CHECK:       .Lfunc_end{{[0-9]+}}:
define <32 x float> @fmaxnum_v32f32_nnan(<32 x float> %a, <32 x float> %b) {
  %r = call nnan <32 x float> @llvm.maxnum.v32f32(<32 x float> %a, <32 x float> %b)
  ret <32 x float> %r
}

declare float @llvm.vector.reduce.fmin.v32f32(<32 x float>)
declare <32 x float> @llvm.minnum.v32f32(<32 x float>, <32 x float>)
declare float @llvm.vector.reduce.fmax.v32f32(<32 x float>)
declare <32 x float> @llvm.maxnum.v32f32(<32 x float>, <32 x float>)
