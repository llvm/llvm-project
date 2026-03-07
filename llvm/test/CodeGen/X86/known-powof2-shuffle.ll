; RUN: llc -mtriple=x86_64-unknown-linux-gnu -o - %s | FileCheck %s

; Test that isKnownToBeAPowerOfTwo handles VECTOR_SHUFFLE

; A shuffle of two power-of-2 vectors should be recognized as power-of-2
define <4 x i32> @shuffle_pow2(<4 x i32> %a) {
; CHECK-LABEL: shuffle_pow2:
  %pow2 = and <4 x i32> %a, <i32 4, i32 4, i32 4, i32 4>
  %splat = shufflevector <4 x i32> %pow2, <4 x i32> poison,
                         <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  ret <4 x i32> %splat
}
