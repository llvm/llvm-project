; RUN: llc -O2 -mtriple=nvptx64 -mcpu=sm_90 %s -o - | FileCheck %s
; RUN: %if ptxas-sm_90 && ptxas-isa-7.8 %{ llc -O2 -mtriple=nvptx64 -mcpu=sm_90 %s | %ptxas-verify -arch=sm_90 %}

; LowerShift{Right,Left}Parts build the carry-in half of the result with a plain
; ISD::SHL/SRL by (element width - Amt). PTX clamps shift amounts, so that is
; fine at run time, but when Amt is the constant 0 the DAG sees a shift by the
; full element width, which is poison: getNode folded it to undef, the following
; OR then produced the constant -1, and the round-trip comparison of a
; saturating shift compared against -1 instead of the shifted value.
;
; The pdep mask only has bit 96 set, so the low 64 bits of the shift amount are
; provably zero while the amount itself stays data-dependent. Reverting either
; half of the fix makes both functions below materialize the bogus constant
; again.
;
; https://github.com/llvm/llvm-project/issues/244584

define <1 x i127> @ushl_sat_zero_low_amount(<1 x i127> %b) {
; CHECK-LABEL: ushl_sat_zero_low_amount(
; A shift of the high half by the full element width must not be folded into a
; constant: this mov is that folded value.
; CHECK-NOT: {{mov\.b64.*-1;}}
; CHECK: ret
  %v0 = call <1 x i127> @llvm.pdep.v1i127(<1 x i127> %b, <1 x i127> <i127 79228162514264337593543950336>)
  %v1 = call <1 x i127> @llvm.ushl.sat.v1i127(<1 x i127> %b, <1 x i127> %v0)
  ret <1 x i127> %v1
}

define <1 x i127> @lshr_of_sat_zero_low_amount(<1 x i127> %b) {
; CHECK-LABEL: lshr_of_sat_zero_low_amount(
; CHECK-NOT: {{mov\.b64.*-1;}}
; CHECK: ret
  %v0 = call <1 x i127> @llvm.pdep.v1i127(<1 x i127> %b, <1 x i127> <i127 79228162514264337593543950336>)
  %s = call <1 x i127> @llvm.ushl.sat.v1i127(<1 x i127> %b, <1 x i127> %v0)
  %r = lshr <1 x i127> %s, %v0
  ret <1 x i127> %r
}

declare <1 x i127> @llvm.pdep.v1i127(<1 x i127>, <1 x i127>)
declare <1 x i127> @llvm.ushl.sat.v1i127(<1 x i127>, <1 x i127>)
