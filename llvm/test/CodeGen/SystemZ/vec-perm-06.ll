; Test vector pack.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a canonical v16i8 pack.
define <16 x i8> @f1(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vpkh %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 1, i32 3, i32 5, i32 7,
                                   i32 9, i32 11, i32 13, i32 15,
                                   i32 17, i32 19, i32 21, i32 23,
                                   i32 25, i32 27, i32 29, i32 31>
  ret <16 x i8> %ret
}

; Test a reversed v16i8 pack.
define <16 x i8> @f2(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f2:
; CHECK: vpkh %v24, %v26, %v24
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 17, i32 19, i32 21, i32 23,
                                   i32 25, i32 27, i32 29, i32 31,
                                   i32 1, i32 3, i32 5, i32 7,
                                   i32 9, i32 11, i32 13, i32 15>
  ret <16 x i8> %ret
}

; Test a v16i8 pack with only the first operand being used.
define <16 x i8> @f3(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f3:
; CHECK: vpkh %v24, %v24, %v24
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 1, i32 3, i32 5, i32 7,
                                   i32 9, i32 11, i32 13, i32 15,
                                   i32 1, i32 3, i32 5, i32 7,
                                   i32 9, i32 11, i32 13, i32 15>
  ret <16 x i8> %ret
}

; Test a v16i8 pack with only the second operand being used.
; This is converted into @f3 by target-independent code.
define <16 x i8> @f4(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f4:
; CHECK: vpkh %v24, %v26, %v26
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 17, i32 19, i32 21, i32 23,
                                   i32 25, i32 27, i32 29, i32 31,
                                   i32 17, i32 19, i32 21, i32 23,
                                   i32 25, i32 27, i32 29, i32 31>
  ret <16 x i8> %ret
}

; Test a v16i8 pack with both operands being the same.  This too is
; converted into @f3 by target-independent code.
define <16 x i8> @f5(<16 x i8> %val) {
; CHECK-LABEL: f5:
; CHECK: vpkh %v24, %v24, %v24
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val, <16 x i8> %val,
                       <16 x i32> <i32 1, i32 3, i32 5, i32 7,
                                   i32 9, i32 11, i32 13, i32 15,
                                   i32 17, i32 19, i32 21, i32 23,
                                   i32 25, i32 27, i32 29, i32 31>
  ret <16 x i8> %ret
}

; Test a v16i8 pack in which some of the indices are don't care.
define <16 x i8> @f6(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f6:
; CHECK: vpkh %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 1, i32 undef, i32 5, i32 7,
                                   i32 undef, i32 11, i32 undef, i32 undef,
                                   i32 undef, i32 19, i32 21, i32 23,
                                   i32 undef, i32 27, i32 29, i32 undef>
  ret <16 x i8> %ret
}

; Test a v16i8 pack in which one of the operands is undefined and where
; indices for that operand are "don't care".  Target-independent code
; converts the indices themselves into "undef"s.
define <16 x i8> @f7(<16 x i8> %val) {
; CHECK-LABEL: f7:
; CHECK: vpkh %v24, %v24, %v24
; CHECK: br %r14
  %ret = shufflevector <16 x i8> undef, <16 x i8> %val,
                       <16 x i32> <i32 7, i32 1, i32 9, i32 15,
                                   i32 15, i32 3, i32 5, i32 1,
                                   i32 17, i32 19, i32 21, i32 23,
                                   i32 25, i32 27, i32 29, i32 31>
  ret <16 x i8> %ret
}

; Test a canonical v8i16 pack.
define <8 x i16> @f8(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f8:
; CHECK: vpkf %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <8 x i16> %val1, <8 x i16> %val2,
                       <8 x i32> <i32 1, i32 3, i32 5, i32 7,
                                  i32 9, i32 11, i32 13, i32 15>
  ret <8 x i16> %ret
}

; Test a reversed v8i16 pack.
define <8 x i16> @f9(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f9:
; CHECK: vpkf %v24, %v26, %v24
; CHECK: br %r14
  %ret = shufflevector <8 x i16> %val1, <8 x i16> %val2,
                       <8 x i32> <i32 9, i32 11, i32 13, i32 15,
                                  i32 1, i32 3, i32 5, i32 7>
  ret <8 x i16> %ret
}

; Test a canonical v4i32 pack.
define <4 x i32> @f10(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f10:
; CHECK: vpkg %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <4 x i32> %val1, <4 x i32> %val2,
                       <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret <4 x i32> %ret
}

; Test a reversed v4i32 pack.
define <4 x i32> @f11(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f11:
; CHECK: vpkg %v24, %v26, %v24
; CHECK: br %r14
  %ret = shufflevector <4 x i32> %val1, <4 x i32> %val2,
                       <4 x i32> <i32 5, i32 7, i32 1, i32 3>
  ret <4 x i32> %ret
}

; Test a canonical v4f32 pack.
define <4 x float> @f12(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f12:
; CHECK: vpkg %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <4 x float> %val1, <4 x float> %val2,
                       <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret <4 x float> %ret
}

; Test a reversed v4f32 pack.
define <4 x float> @f13(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f13:
; CHECK: vpkg %v24, %v26, %v24
; CHECK: br %r14
  %ret = shufflevector <4 x float> %val1, <4 x float> %val2,
                       <4 x i32> <i32 5, i32 7, i32 1, i32 3>
  ret <4 x float> %ret
}
