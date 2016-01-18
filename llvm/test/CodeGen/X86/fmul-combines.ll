; RUN: llc -mtriple=x86_64-unknown-unknown -march=x86-64 < %s | FileCheck %s

; CHECK-LABEL: fmul2_f32:
; CHECK: addss %xmm0, %xmm0
define float @fmul2_f32(float %x) {
  %y = fmul float %x, 2.0
  ret float %y
}

; fmul 2.0, x -> fadd x, x for vectors.

; CHECK-LABEL: fmul2_v4f32:
; CHECK: addps %xmm0, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fmul2_v4f32(<4 x float> %x) {
  %y = fmul <4 x float> %x, <float 2.0, float 2.0, float 2.0, float 2.0>
  ret <4 x float> %y
}

; CHECK-LABEL: constant_fold_fmul_v4f32:
; CHECK: movaps
; CHECK-NEXT: ret
define <4 x float> @constant_fold_fmul_v4f32(<4 x float> %x) {
  %y = fmul <4 x float> <float 4.0, float 4.0, float 4.0, float 4.0>, <float 2.0, float 2.0, float 2.0, float 2.0>
  ret <4 x float> %y
}

; CHECK-LABEL: fmul0_v4f32:
; CHECK: xorps %xmm0, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fmul0_v4f32(<4 x float> %x) #0 {
  %y = fmul <4 x float> %x, <float 0.0, float 0.0, float 0.0, float 0.0>
  ret <4 x float> %y
}

; CHECK-LABEL: fmul_c2_c4_v4f32:
; CHECK-NOT: addps
; CHECK: mulps
; CHECK-NOT: mulps
; CHECK-NEXT: ret
define <4 x float> @fmul_c2_c4_v4f32(<4 x float> %x) #0 {
  %y = fmul <4 x float> %x, <float 2.0, float 2.0, float 2.0, float 2.0>
  %z = fmul <4 x float> %y, <float 4.0, float 4.0, float 4.0, float 4.0>
  ret <4 x float> %z
}

; CHECK-LABEL: fmul_c3_c4_v4f32:
; CHECK-NOT: addps
; CHECK: mulps
; CHECK-NOT: mulps
; CHECK-NEXT: ret
define <4 x float> @fmul_c3_c4_v4f32(<4 x float> %x) #0 {
  %y = fmul <4 x float> %x, <float 3.0, float 3.0, float 3.0, float 3.0>
  %z = fmul <4 x float> %y, <float 4.0, float 4.0, float 4.0, float 4.0>
  ret <4 x float> %z
}

; We should be able to pre-multiply the two constant vectors.
; CHECK: float 5.000000e+00
; CHECK: float 1.200000e+01
; CHECK: float 2.100000e+01
; CHECK: float 3.200000e+01
; CHECK-LABEL: fmul_v4f32_two_consts_no_splat:
; CHECK: mulps
; CHECK-NOT: mulps
; CHECK-NEXT: ret
define <4 x float> @fmul_v4f32_two_consts_no_splat(<4 x float> %x) #0 {
  %y = fmul <4 x float> %x, <float 1.0, float 2.0, float 3.0, float 4.0>
  %z = fmul <4 x float> %y, <float 5.0, float 6.0, float 7.0, float 8.0>
  ret <4 x float> %z
}

; Same as above, but reverse operands to make sure non-canonical form is also handled.
; CHECK: float 5.000000e+00
; CHECK: float 1.200000e+01
; CHECK: float 2.100000e+01
; CHECK: float 3.200000e+01
; CHECK-LABEL: fmul_v4f32_two_consts_no_splat_non_canonical:
; CHECK: mulps
; CHECK-NOT: mulps
; CHECK-NEXT: ret
define <4 x float> @fmul_v4f32_two_consts_no_splat_non_canonical(<4 x float> %x) #0 {
  %y = fmul <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, %x
  %z = fmul <4 x float> <float 5.0, float 6.0, float 7.0, float 8.0>, %y
  ret <4 x float> %z
}

; More than one use of a constant multiply should not inhibit the optimization.
; Instead of a chain of 2 dependent mults, this test will have 2 independent mults.
; CHECK: float 6.000000e+00
; CHECK: float 1.400000e+01
; CHECK: float 2.400000e+01
; CHECK: float 3.600000e+01
; CHECK-LABEL: fmul_v4f32_two_consts_no_splat_multiple_use:
; CHECK: mulps
; CHECK: ret
define <4 x float> @fmul_v4f32_two_consts_no_splat_multiple_use(<4 x float> %x) #0 {
  %y = fmul <4 x float> %x, <float 1.0, float 2.0, float 3.0, float 4.0>
  %z = fmul <4 x float> %y, <float 5.0, float 6.0, float 7.0, float 8.0>
  %a = fadd <4 x float> %y, %z
  ret <4 x float> %a
}

; PR22698 - http://llvm.org/bugs/show_bug.cgi?id=22698
; Make sure that we don't infinite loop swapping constants back and forth.

define <4 x float> @PR22698_splats(<4 x float> %a) #0 {
  %mul1 = fmul fast <4 x float> <float 2.0, float 2.0, float 2.0, float 2.0>, <float 3.0, float 3.0, float 3.0, float 3.0>
  %mul2 = fmul fast <4 x float> <float 4.0, float 4.0, float 4.0, float 4.0>, %mul1
  %mul3 = fmul fast <4 x float> %a, %mul2
  ret <4 x float> %mul3

; CHECK: float 2.400000e+01
; CHECK: float 2.400000e+01
; CHECK: float 2.400000e+01
; CHECK: float 2.400000e+01
; CHECK-LABEL: PR22698_splats:
; CHECK: mulps
; CHECK: ret
}

; Same as above, but verify that non-splat vectors are handled correctly too.
define <4 x float> @PR22698_no_splats(<4 x float> %a) #0 {
  %mul1 = fmul fast <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, <float 5.0, float 6.0, float 7.0, float 8.0>
  %mul2 = fmul fast <4 x float> <float 9.0, float 10.0, float 11.0, float 12.0>, %mul1
  %mul3 = fmul fast <4 x float> %a, %mul2
  ret <4 x float> %mul3

; CHECK: float 4.500000e+01
; CHECK: float 1.200000e+02
; CHECK: float 2.310000e+02
; CHECK: float 3.840000e+02
; CHECK-LABEL: PR22698_no_splats:
; CHECK: mulps
; CHECK: ret
}

; CHECK-LABEL: fmul_c2_c4_f32:
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: mulss
; CHECK-NEXT: ret
define float @fmul_c2_c4_f32(float %x) #0 {
  %y = fmul float %x, 2.0
  %z = fmul float %y, 4.0
  ret float %z
}

; CHECK-LABEL: fmul_c3_c4_f32:
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: mulss
; CHECK-NET: ret
define float @fmul_c3_c4_f32(float %x) #0 {
  %y = fmul float %x, 3.0
  %z = fmul float %y, 4.0
  ret float %z
}

; CHECK-LABEL: fmul_fneg_fneg_f32:
; CHECK: mulss %xmm1, %xmm0
; CHECK-NEXT: retq
define float @fmul_fneg_fneg_f32(float %x, float %y) {
  %x.neg = fsub float -0.0, %x
  %y.neg = fsub float -0.0, %y
  %mul = fmul float %x.neg, %y.neg
  ret float %mul
}
; CHECK-LABEL: fmul_fneg_fneg_v4f32:
; CHECK: mulps {{%xmm1|\(%rdx\)}}, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fmul_fneg_fneg_v4f32(<4 x float> %x, <4 x float> %y) {
  %x.neg = fsub <4 x float> <float -0.0, float -0.0, float -0.0, float -0.0>, %x
  %y.neg = fsub <4 x float> <float -0.0, float -0.0, float -0.0, float -0.0>, %y
  %mul = fmul <4 x float> %x.neg, %y.neg
  ret <4 x float> %mul
}

attributes #0 = { "less-precise-fpmad"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "unsafe-fp-math"="true" }
