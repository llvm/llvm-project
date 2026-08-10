; Test vector merge high.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a canonical v16i8 merge high.
define <16 x i8> @f1(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vmrhb %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 0, i32 16, i32 1, i32 17,
                                   i32 2, i32 18, i32 3, i32 19,
                                   i32 4, i32 20, i32 5, i32 21,
                                   i32 6, i32 22, i32 7, i32 23>
  ret <16 x i8> %ret
}

; Test a reversed v16i8 merge high.
define <16 x i8> @f2(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f2:
; CHECK: vmrhb %v24, %v26, %v24
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 16, i32 0, i32 17, i32 1,
                                   i32 18, i32 2, i32 19, i32 3,
                                   i32 20, i32 4, i32 21, i32 5,
                                   i32 22, i32 6, i32 23, i32 7>
  ret <16 x i8> %ret
}

; Test a v16i8 merge high with only the first operand being used.
define <16 x i8> @f3(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f3:
; CHECK: vmrhb %v24, %v24, %v24
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 0, i32 0, i32 1, i32 1,
                                   i32 2, i32 2, i32 3, i32 3,
                                   i32 4, i32 4, i32 5, i32 5,
                                   i32 6, i32 6, i32 7, i32 7>
  ret <16 x i8> %ret
}

; Test a v16i8 merge high with only the second operand being used.
; This is converted into @f3 by target-independent code.
define <16 x i8> @f4(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f4:
; CHECK: vmrhb %v24, %v26, %v26
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 16, i32 16, i32 17, i32 17,
                                   i32 18, i32 18, i32 19, i32 19,
                                   i32 20, i32 20, i32 21, i32 21,
                                   i32 22, i32 22, i32 23, i32 23>
  ret <16 x i8> %ret
}

; Test a v16i8 merge with both operands being the same.  This too is
; converted into @f3 by target-independent code.
define <16 x i8> @f5(<16 x i8> %val) {
; CHECK-LABEL: f5:
; CHECK: vmrhb %v24, %v24, %v24
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val, <16 x i8> %val,
                       <16 x i32> <i32 0, i32 16, i32 17, i32 17,
                                   i32 18, i32 2, i32 3, i32 3,
                                   i32 20, i32 20, i32 5, i32 5,
                                   i32 6, i32 22, i32 23, i32 7>
  ret <16 x i8> %ret
}

; Test a v16i8 merge in which some of the indices are don't care.
define <16 x i8> @f6(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f6:
; CHECK: vmrhb %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val1, <16 x i8> %val2,
                       <16 x i32> <i32 0, i32 undef, i32 1, i32 17,
                                   i32 undef, i32 18, i32 undef, i32 undef,
                                   i32 undef, i32 20, i32 5, i32 21,
                                   i32 undef, i32 22, i32 7, i32 undef>
  ret <16 x i8> %ret
}

; Test a v16i8 merge in which one of the operands is undefined and where
; indices for that operand are "don't care".  Target-independent code
; converts the indices themselves into "undef"s.
define <16 x i8> @f7(<16 x i8> %val) {
; CHECK-LABEL: f7:
; CHECK: vmrhb %v24, %v24, %v24
; CHECK: br %r14
  %ret = shufflevector <16 x i8> undef, <16 x i8> %val,
                       <16 x i32> <i32 11, i32 16, i32 17, i32 5,
                                   i32 18, i32 10, i32 19, i32 19,
                                   i32 20, i32 20, i32 21, i32 3,
                                   i32 2, i32 22, i32 9, i32 23>
  ret <16 x i8> %ret
}

; Test a canonical v8i16 merge high.
define <8 x i16> @f8(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f8:
; CHECK: vmrhh %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <8 x i16> %val1, <8 x i16> %val2,
                       <8 x i32> <i32 0, i32 8, i32 1, i32 9,
                                  i32 2, i32 10, i32 3, i32 11>
  ret <8 x i16> %ret
}

; Test a reversed v8i16 merge high.
define <8 x i16> @f9(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f9:
; CHECK: vmrhh %v24, %v26, %v24
; CHECK: br %r14
  %ret = shufflevector <8 x i16> %val1, <8 x i16> %val2,
                       <8 x i32> <i32 8, i32 0, i32 9, i32 1,
                                  i32 10, i32 2, i32 11, i32 3>
  ret <8 x i16> %ret
}

; Test a canonical v4i32 merge high.
define <4 x i32> @f10(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f10:
; CHECK: vmrhf %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <4 x i32> %val1, <4 x i32> %val2,
                       <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i32> %ret
}

; Test a reversed v4i32 merge high.
define <4 x i32> @f11(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f11:
; CHECK: vmrhf %v24, %v26, %v24
; CHECK: br %r14
  %ret = shufflevector <4 x i32> %val1, <4 x i32> %val2,
                       <4 x i32> <i32 4, i32 0, i32 5, i32 1>
  ret <4 x i32> %ret
}

; Test a canonical v2i64 merge high.
define <2 x i64> @f12(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f12:
; CHECK: vmrhg %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <2 x i64> %val1, <2 x i64> %val2,
                       <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %ret
}

; Test a reversed v2i64 merge high.
define <2 x i64> @f13(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f13:
; CHECK: vmrhg %v24, %v26, %v24
; CHECK: br %r14
  %ret = shufflevector <2 x i64> %val1, <2 x i64> %val2,
                       <2 x i32> <i32 2, i32 0>
  ret <2 x i64> %ret
}

; Test a canonical v4f32 merge high.
define <4 x float> @f14(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f14:
; CHECK: vmrhf %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <4 x float> %val1, <4 x float> %val2,
                       <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x float> %ret
}

; Test a reversed v4f32 merge high.
define <4 x float> @f15(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f15:
; CHECK: vmrhf %v24, %v26, %v24
; CHECK: br %r14
  %ret = shufflevector <4 x float> %val1, <4 x float> %val2,
                       <4 x i32> <i32 4, i32 0, i32 5, i32 1>
  ret <4 x float> %ret
}

; Test a canonical v2f64 merge high.
define <2 x double> @f16(<2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f16:
; CHECK: vmrhg %v24, %v24, %v26
; CHECK: br %r14
  %ret = shufflevector <2 x double> %val1, <2 x double> %val2,
                       <2 x i32> <i32 0, i32 2>
  ret <2 x double> %ret
}

; Test a reversed v2f64 merge high.
define <2 x double> @f17(<2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f17:
; CHECK: vmrhg %v24, %v26, %v24
; CHECK: br %r14
  %ret = shufflevector <2 x double> %val1, <2 x double> %val2,
                       <2 x i32> <i32 2, i32 0>
  ret <2 x double> %ret
}
