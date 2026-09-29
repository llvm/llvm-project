; Test vector AND.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 AND.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vn %v24, %v26, %v28
; CHECK: br %r14
  %ret = and <16 x i8> %val1, %val2
  ret <16 x i8> %ret
}

; Test a v8i16 AND.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f2:
; CHECK: vn %v24, %v26, %v28
; CHECK: br %r14
  %ret = and <8 x i16> %val1, %val2
  ret <8 x i16> %ret
}

; Test a v4i32 AND.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vn %v24, %v26, %v28
; CHECK: br %r14
  %ret = and <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v2i64 AND.
define <2 x i64> @f4(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f4:
; CHECK: vn %v24, %v26, %v28
; CHECK: br %r14
  %ret = and <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}
