; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s

; Numbers smaller than -127 and greater than or equal to 127 are not allowed.
; This should get lowered to a regular vector multiply and these tests should
; be updated when those patterns are added.

define <vscale x 2 x i64> @mul_i64_neg_1(<vscale x 2 x i64> %a) {
  %elt = insertelement <vscale x 2 x i64> undef, i64 255, i32 0
  %splat = shufflevector <vscale x 2 x i64> %elt, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res = mul <vscale x 2 x i64> %a, %splat
  ret <vscale x 2 x i64> %res
}
