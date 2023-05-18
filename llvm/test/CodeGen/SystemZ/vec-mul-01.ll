; Test vector multiplication.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 multiplication.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vmlb %v24, %v26, %v28
; CHECK: br %r14
  %ret = mul <16 x i8> %val1, %val2
  ret <16 x i8> %ret
}

; Test a v8i16 multiplication.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f2:
; CHECK: vmlhw %v24, %v26, %v28
; CHECK: br %r14
  %ret = mul <8 x i16> %val1, %val2
  ret <8 x i16> %ret
}

; Test a v4i32 multiplication.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vmlf %v24, %v26, %v28
; CHECK: br %r14
  %ret = mul <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v2i64 multiplication.  There's no vector equivalent.
define <2 x i64> @f4(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f4:
; CHECK-NOT: vmlg
; CHECK: br %r14
  %ret = mul <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}

; Test a v2f64 multiplication.
define <2 x double> @f5(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) {
; CHECK-LABEL: f5:
; CHECK: vfmdb %v24, %v26, %v28
; CHECK: br %r14
  %ret = fmul <2 x double> %val1, %val2
  ret <2 x double> %ret
}

; Test an f64 multiplication that uses vector registers.
define double @f6(<2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f6:
; CHECK: wfmdb %f0, %v24, %v26
; CHECK: br %r14
  %scalar1 = extractelement <2 x double> %val1, i32 0
  %scalar2 = extractelement <2 x double> %val2, i32 0
  %ret = fmul double %scalar1, %scalar2
  ret double %ret
}
