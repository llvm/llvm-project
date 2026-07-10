; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK-LABEL: @dynamic(
; CHECK: %r = shufflevector <4 x i32> %a, <4 x i32> %b, <8 x i8> %mask
define <8 x i32> @dynamic(<4 x i32> %a, <4 x i32> %b, <8 x i8> %mask) {
  %r = shufflevector <4 x i32> %a, <4 x i32> %b, <8 x i8> %mask
  ret <8 x i32> %r
}

; Constant masks print exactly as before.
; CHECK-LABEL: @constant(
; CHECK: %r = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 poison, i32 7>
define <4 x i32> @constant(<4 x i32> %a, <4 x i32> %b) {
  %r = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 poison, i32 7>
  ret <4 x i32> %r
}

; A non-canonical (out-of-bounds) constant mask is now valid: OOB lanes are poison.
; CHECK-LABEL: @oob_constant(
; CHECK: %r = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 99, i32 1, i32 2>
define <4 x i32> @oob_constant(<4 x i32> %a, <4 x i32> %b) {
  %r = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 99, i32 1, i32 2>
  ret <4 x i32> %r
}

; Scalable dynamic mask.
; CHECK-LABEL: @scalable(
; CHECK: %r = shufflevector <vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x i16> %mask
define <vscale x 4 x float> @scalable(<vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x i16> %mask) {
  %r = shufflevector <vscale x 4 x float> %a, <vscale x 4 x float> %b, <vscale x 4 x i16> %mask
  ret <vscale x 4 x float> %r
}
