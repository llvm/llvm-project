; RUN: not opt -passes=verify -S < %s 2>&1 | FileCheck %s

; CHECK: Mask must be a vector of integers.
define <4 x i32> @mask_not_integer(<4 x i32> %v, <4 x float> %mask) {
  %res = call <4 x i32> @llvm.vector.shuffle.v4i32.v4f32(<4 x i32> %v, <4 x float> %mask)
  ret <4 x i32> %res
}

; CHECK: Mask and return type must have the same number of elements.
define <4 x i32> @mask_too_long(<4 x i32> %v, <8 x i8> %mask) {
  %res = call <4 x i32> @llvm.vector.shuffle.v4i32.v8i8(<4 x i32> %v, <8 x i8> %mask)
  ret <4 x i32> %res
}

; CHECK: Mask and return type must have the same number of elements.
define <4 x i32> @mask_too_short(<4 x i32> %v, <2 x i8> %mask) {
  %res = call <4 x i32> @llvm.vector.shuffle.v4i32.v2i8(<4 x i32> %v, <2 x i8> %mask)
  ret <4 x i32> %res
}

; A fixed-length mask never has the same element count as a scalable result.
; CHECK: Mask and return type must have the same number of elements.
define <vscale x 4 x i32> @mixed_scalable(<vscale x 4 x i32> %v, <4 x i8> %mask) {
  %res = call <vscale x 4 x i32> @llvm.vector.shuffle.nxv4i32.v4i8(<vscale x 4 x i32> %v, <4 x i8> %mask)
  ret <vscale x 4 x i32> %res
}
