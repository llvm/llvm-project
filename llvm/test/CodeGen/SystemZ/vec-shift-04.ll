; Test vector shift left with scalar shift amount.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 shift by a variable.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, i32 %shift) {
; CHECK-LABEL: f1:
; CHECK: veslb %v24, %v26, 0(%r2)
; CHECK: br %r14
  %truncshift = trunc i32 %shift to i8
  %shiftvec = insertelement <16 x i8> undef, i8 %truncshift, i32 0
  %val2 = shufflevector <16 x i8> %shiftvec, <16 x i8> undef,
                        <16 x i32> zeroinitializer
  %ret = shl <16 x i8> %val1, %val2
  ret <16 x i8> %ret
}

; Test a v16i8 shift by the lowest useful constant.
define <16 x i8> @f2(<16 x i8> %dummy, <16 x i8> %val) {
; CHECK-LABEL: f2:
; CHECK: veslb %v24, %v26, 1
; CHECK: br %r14
  %ret = shl <16 x i8> %val, <i8 1, i8 1, i8 1, i8 1,
                              i8 1, i8 1, i8 1, i8 1,
                              i8 1, i8 1, i8 1, i8 1,
                              i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %ret
}

; Test a v16i8 shift by the highest useful constant.
define <16 x i8> @f3(<16 x i8> %dummy, <16 x i8> %val) {
; CHECK-LABEL: f3:
; CHECK: veslb %v24, %v26, 7
; CHECK: br %r14
  %ret = shl <16 x i8> %val, <i8 7, i8 7, i8 7, i8 7,
                              i8 7, i8 7, i8 7, i8 7,
                              i8 7, i8 7, i8 7, i8 7,
                              i8 7, i8 7, i8 7, i8 7>
  ret <16 x i8> %ret
}

; Test a v8i16 shift by a variable.
define <8 x i16> @f4(<8 x i16> %dummy, <8 x i16> %val1, i32 %shift) {
; CHECK-LABEL: f4:
; CHECK: veslh %v24, %v26, 0(%r2)
; CHECK: br %r14
  %truncshift = trunc i32 %shift to i16
  %shiftvec = insertelement <8 x i16> undef, i16 %truncshift, i32 0
  %val2 = shufflevector <8 x i16> %shiftvec, <8 x i16> undef,
                        <8 x i32> zeroinitializer
  %ret = shl <8 x i16> %val1, %val2
  ret <8 x i16> %ret
}

; Test a v8i16 shift by the lowest useful constant.
define <8 x i16> @f5(<8 x i16> %dummy, <8 x i16> %val) {
; CHECK-LABEL: f5:
; CHECK: veslh %v24, %v26, 1
; CHECK: br %r14
  %ret = shl <8 x i16> %val, <i16 1, i16 1, i16 1, i16 1,
                              i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %ret
}

; Test a v8i16 shift by the highest useful constant.
define <8 x i16> @f6(<8 x i16> %dummy, <8 x i16> %val) {
; CHECK-LABEL: f6:
; CHECK: veslh %v24, %v26, 15
; CHECK: br %r14
  %ret = shl <8 x i16> %val, <i16 15, i16 15, i16 15, i16 15,
                              i16 15, i16 15, i16 15, i16 15>
  ret <8 x i16> %ret
}

; Test a v4i32 shift by a variable.
define <4 x i32> @f7(<4 x i32> %dummy, <4 x i32> %val1, i32 %shift) {
; CHECK-LABEL: f7:
; CHECK: veslf %v24, %v26, 0(%r2)
; CHECK: br %r14
  %shiftvec = insertelement <4 x i32> undef, i32 %shift, i32 0
  %val2 = shufflevector <4 x i32> %shiftvec, <4 x i32> undef,
                        <4 x i32> zeroinitializer
  %ret = shl <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v4i32 shift by the lowest useful constant.
define <4 x i32> @f8(<4 x i32> %dummy, <4 x i32> %val) {
; CHECK-LABEL: f8:
; CHECK: veslf %v24, %v26, 1
; CHECK: br %r14
  %ret = shl <4 x i32> %val, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %ret
}

; Test a v4i32 shift by the highest useful constant.
define <4 x i32> @f9(<4 x i32> %dummy, <4 x i32> %val) {
; CHECK-LABEL: f9:
; CHECK: veslf %v24, %v26, 31
; CHECK: br %r14
  %ret = shl <4 x i32> %val, <i32 31, i32 31, i32 31, i32 31>
  ret <4 x i32> %ret
}

; Test a v2i64 shift by a variable.
define <2 x i64> @f10(<2 x i64> %dummy, <2 x i64> %val1, i32 %shift) {
; CHECK-LABEL: f10:
; CHECK: veslg %v24, %v26, 0(%r2)
; CHECK: br %r14
  %extshift = sext i32 %shift to i64
  %shiftvec = insertelement <2 x i64> undef, i64 %extshift, i32 0
  %val2 = shufflevector <2 x i64> %shiftvec, <2 x i64> undef,
                        <2 x i32> zeroinitializer
  %ret = shl <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}

; Test a v2i64 shift by the lowest useful constant.
define <2 x i64> @f11(<2 x i64> %dummy, <2 x i64> %val) {
; CHECK-LABEL: f11:
; CHECK: veslg %v24, %v26, 1
; CHECK: br %r14
  %ret = shl <2 x i64> %val, <i64 1, i64 1>
  ret <2 x i64> %ret
}

; Test a v2i64 shift by the highest useful constant.
define <2 x i64> @f12(<2 x i64> %dummy, <2 x i64> %val) {
; CHECK-LABEL: f12:
; CHECK: veslg %v24, %v26, 63
; CHECK: br %r14
  %ret = shl <2 x i64> %val, <i64 63, i64 63>
  ret <2 x i64> %ret
}
