; Test vector permutes using VPDI.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a high1/low2 permute for v16i8.
define <16 x i8> @f1(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vpdi %v24, %v24, %v26, 1
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 0, i32 1, i32 2, i32 3,
                                   i32 4, i32 5, i32 6, i32 7,
                                   i32 24, i32 25, i32 26, i32 27,
                                   i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %ret
}

; Test a low2/high1 permute for v16i8.
define <16 x i8> @f2(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f2:
; CHECK: vpdi %v24, %v26, %v24, 4
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 24, i32 25, i32 26, i32 27,
                                   i32 28, i32 29, i32 30, i32 31,
                                   i32 0, i32 1, i32 2, i32 3,
                                   i32 4, i32 5, i32 6, i32 7>
  ret <16 x i8> %ret
}

; Test a low1/high2 permute for v16i8.
define <16 x i8> @f3(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f3:
; CHECK: vpdi %v24, %v24, %v26, 4
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 8, i32 9, i32 10, i32 undef,
                                   i32 12, i32 undef, i32 14, i32 15,
                                   i32 16, i32 17, i32 undef, i32 19,
                                   i32 20, i32 21, i32 22, i32 undef>
  ret <16 x i8> %ret
}

; Test a high2/low1 permute for v16i8.
define <16 x i8> @f4(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f4:
; CHECK: vpdi %v24, %v26, %v24, 1
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 16, i32 17, i32 18, i32 19,
                                   i32 20, i32 21, i32 22, i32 23,
                                   i32 8, i32 9, i32 10, i32 11,
                                   i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %ret
}

; Test reversing two doublewords in a v16i8.
define <16 x i8> @f5(<16 x i8> %val) {
; CHECK-LABEL: f5:
; CHECK: vpdi %v24, %v24, %v24, 4
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val, <16 x i8> undef,
                       <16 x i32> <i32 8, i32 9, i32 10, i32 11,
                                   i32 12, i32 13, i32 14, i32 15,
                                   i32 0, i32 1, i32 2, i32 3,
                                   i32 4, i32 5, i32 6, i32 7>
  ret <16 x i8> %ret
}

; Test a high1/low2 permute for v8i16.
define <8 x i16> @f6(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f6:
; CHECK: vpdi %v24, %v24, %v26, 1
; CHECK: br %r14
  %ret = shufflevector <8 x i16> %val1, <8 x i16> %val2,
                       <8 x i32> <i32 0, i32 1, i32 2, i32 3,
                                  i32 12, i32 13, i32 14, i32 15>
  ret <8 x i16> %ret
}

; Test a low2/high1 permute for v8i16.
define <8 x i16> @f7(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f7:
; CHECK: vpdi %v24, %v26, %v24, 4
; CHECK: br %r14
  %ret = shufflevector <8 x i16> %val1, <8 x i16> %val2,
                       <8 x i32> <i32 12, i32 13, i32 14, i32 15,
                                  i32 0, i32 1, i32 2, i32 3>
  ret <8 x i16> %ret
}

; Test a high1/low2 permute for v4i32.
define <4 x i32> @f8(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f8:
; CHECK: vpdi %v24, %v24, %v26, 1
; CHECK: br %r14
  %ret = shufflevector <4 x i32> %val1, <4 x i32> %val2,
                       <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  ret <4 x i32> %ret
}

; Test a low2/high1 permute for v4i32.
define <4 x i32> @f9(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f9:
; CHECK: vpdi %v24, %v26, %v24, 4
; CHECK: br %r14
  %ret = shufflevector <4 x i32> %val1, <4 x i32> %val2,
                       <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x i32> %ret
}

; Test a high1/low2 permute for v2i64.
define <2 x i64> @f10(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f10:
; CHECK: vpdi %v24, %v24, %v26, 1
; CHECK: br %r14
  %ret = shufflevector <2 x i64> %val1, <2 x i64> %val2,
                       <2 x i32> <i32 0, i32 3>
  ret <2 x i64> %ret
}

; Test low2/high1 permute for v2i64.
define <2 x i64> @f11(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f11:
; CHECK: vpdi %v24, %v26, %v24, 4
; CHECK: br %r14
  %ret = shufflevector <2 x i64> %val1, <2 x i64> %val2,
                       <2 x i32> <i32 3, i32 0>
  ret <2 x i64> %ret
}

; Test a high1/low2 permute for v4f32.
define <4 x float> @f12(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f12:
; CHECK: vpdi %v24, %v24, %v26, 1
; CHECK: br %r14
  %ret = shufflevector <4 x float> %val1, <4 x float> %val2,
                       <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  ret <4 x float> %ret
}

; Test a low2/high1 permute for v4f32.
define <4 x float> @f13(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f13:
; CHECK: vpdi %v24, %v26, %v24, 4
; CHECK: br %r14
  %ret = shufflevector <4 x float> %val1, <4 x float> %val2,
                       <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x float> %ret
}

; Test a high1/low2 permute for v2f64.
define <2 x double> @f14(<2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f14:
; CHECK: vpdi %v24, %v24, %v26, 1
; CHECK: br %r14
  %ret = shufflevector <2 x double> %val1, <2 x double> %val2,
                       <2 x i32> <i32 0, i32 3>
  ret <2 x double> %ret
}

; Test a low2/high1 permute for v2f64.
define <2 x double> @f15(<2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f15:
; CHECK: vpdi %v24, %v26, %v24, 4
; CHECK: br %r14
  %ret = shufflevector <2 x double> %val1, <2 x double> %val2,
                       <2 x i32> <i32 3, i32 0>
  ret <2 x double> %ret
}
