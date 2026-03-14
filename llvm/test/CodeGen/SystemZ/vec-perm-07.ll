; Test vector shift left double immediate.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 shift with the lowest useful shift amount.
define <16 x i8> @f1(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vsldb %v24, %v24, %v26, 1
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 1, i32 2, i32 3, i32 4,
                                   i32 5, i32 6, i32 7, i32 8,
                                   i32 9, i32 10, i32 11, i32 12,
                                   i32 13, i32 14, i32 15, i32 16>
  ret <16 x i8> %ret
}

; Test a v16i8 shift with the highest shift amount.
define <16 x i8> @f2(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f2:
; CHECK: vsldb %v24, %v24, %v26, 15
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 15, i32 16, i32 17, i32 18,
                                   i32 19, i32 20, i32 21, i32 22,
                                   i32 23, i32 24, i32 25, i32 26,
                                   i32 27, i32 28, i32 29, i32 30>
  ret <16 x i8> %ret
}

; Test a v16i8 shift in which the operands need to be reversed.
define <16 x i8> @f3(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f3:
; CHECK: vsldb %v24, %v26, %v24, 4
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 20, i32 21, i32 22, i32 23,
                                   i32 24, i32 25, i32 26, i32 27,
                                   i32 28, i32 29, i32 30, i32 31,
                                   i32 0, i32 1, i32 2, i32 3>
  ret <16 x i8> %ret
}

; Test a v16i8 shift in which the operands need to be duplicated.
define <16 x i8> @f4(<16 x i8> %val) {
; CHECK-LABEL: f4:
; CHECK: vsldb %v24, %v24, %v24, 7
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val, <16 x i8> undef,
                       <16 x i32> <i32 7, i32 8, i32 9, i32 10,
                                   i32 11, i32 12, i32 13, i32 14,
                                   i32 15, i32 0, i32 1, i32 2,
                                   i32 3, i32 4, i32 5, i32 6>
  ret <16 x i8> %ret
}

; Test a v16i8 shift in which some of the indices are undefs.
define <16 x i8> @f5(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f5:
; CHECK: vsldb %v24, %v24, %v26, 11
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef,
                                   i32 15, i32 16, i32 undef, i32 18,
                                   i32 19, i32 20, i32 21, i32 22,
                                   i32 23, i32 24, i32 25, i32 26>
  ret <16 x i8> %ret
}

; ...and again with reversed operands.
define <16 x i8> @f6(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f6:
; CHECK: vsldb %v24, %v26, %v24, 13
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 undef, i32 undef, i32 31, i32 0,
                                   i32 1, i32 2, i32 3, i32 4,
                                   i32 5, i32 6, i32 7, i32 8,
                                   i32 9, i32 10, i32 11, i32 12>
  ret <16 x i8> %ret
}

; Test a v8i16 shift with the lowest useful shift amount.
define <8 x i16> @f7(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f7:
; CHECK: vsldb %v24, %v24, %v26, 2
; CHECK: br %r14
  %ret = shufflevector <8 x i16> %val1, <8 x i16> %val2,
                       <8 x i32> <i32 1, i32 2, i32 3, i32 4,
                                  i32 5, i32 6, i32 7, i32 8>
  ret <8 x i16> %ret
}

; Test a v8i16 shift with the highest useful shift amount.
define <8 x i16> @f8(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f8:
; CHECK: vsldb %v24, %v24, %v26, 14
; CHECK: br %r14
  %ret = shufflevector <8 x i16> %val1, <8 x i16> %val2,
                       <8 x i32> <i32 7, i32 8, i32 9, i32 10,
                                  i32 11, i32 12, i32 13, i32 14>
  ret <8 x i16> %ret
}

; Test a v4i32 shift with the lowest useful shift amount.
define <4 x i32> @f9(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f9:
; CHECK: vsldb %v24, %v24, %v26, 4
; CHECK: br %r14
  %ret = shufflevector <4 x i32> %val1, <4 x i32> %val2,
                       <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i32> %ret
}

; Test a v4i32 shift with the highest useful shift amount.
define <4 x i32> @f10(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f10:
; CHECK: vsldb %v24, %v24, %v26, 12
; CHECK: br %r14
  %ret = shufflevector <4 x i32> %val1, <4 x i32> %val2,
                       <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x i32> %ret
}

; Test a v4f32 shift with the lowest useful shift amount.
define <4 x float> @f12(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f12:
; CHECK: vsldb %v24, %v24, %v26, 4
; CHECK: br %r14
  %ret = shufflevector <4 x float> %val1, <4 x float> %val2,
                       <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x float> %ret
}

; Test a v4f32 shift with the highest useful shift amount.
define <4 x float> @f13(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f13:
; CHECK: vsldb %v24, %v24, %v26, 12
; CHECK: br %r14
  %ret = shufflevector <4 x float> %val1, <4 x float> %val2,
                       <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x float> %ret
}

; We use VPDI for v2i64 shuffles.
