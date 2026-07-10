; RUN: opt -passes=complex-deinterleaving -mtriple=aarch64-linux-gnu -mattr=+sve -S -disable-output < %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -O2 -o /dev/null %s
; The complex deinterleaving pass (and AArch64 codegen for fixed-width
; vectors) must not crash on shufflevectors with dynamic masks. This test
; passes if neither command crashes.

; Looks like a complex multiply built from deinterleaving shuffles, except the
; masks are run-time values.
define <4 x float> @dynamic_deinterleave(<8 x float> %a, <8 x i32> %m, <4 x i32> %m2) {
entry:
  %ar = shufflevector <8 x float> %a, <8 x float> poison, <4 x i32> %m2
  %ai = shufflevector <8 x float> %a, <8 x float> poison, <4 x i32> %m2
  %mul = fmul fast <4 x float> %ar, %ai
  ret <4 x float> %mul
}

; A dynamic interleaving-shaped root shuffle.
define <8 x float> @dynamic_interleave(<4 x float> %r, <4 x float> %i, <8 x i32> %m) {
entry:
  %s = shufflevector <4 x float> %r, <4 x float> %i, <8 x i32> %m
  ret <8 x float> %s
}
