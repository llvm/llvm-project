; Test vector addition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 addition.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vab %v24, %v26, %v28
; CHECK: br %r14
  %ret = add <16 x i8> %val1, %val2
  ret <16 x i8> %ret
}

; Test a v8i16 addition.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f2:
; CHECK: vah %v24, %v26, %v28
; CHECK: br %r14
  %ret = add <8 x i16> %val1, %val2
  ret <8 x i16> %ret
}

; Test a v4i32 addition.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vaf %v24, %v26, %v28
; CHECK: br %r14
  %ret = add <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v2i64 addition.
define <2 x i64> @f4(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f4:
; CHECK: vag %v24, %v26, %v28
; CHECK: br %r14
  %ret = add <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}

; Test a v2f64 addition.
define <2 x double> @f5(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) {
; CHECK-LABEL: f5:
; CHECK: vfadb %v24, %v26, %v28
; CHECK: br %r14
  %ret = fadd <2 x double> %val1, %val2
  ret <2 x double> %ret
}

; Test an f64 addition that uses vector registers.
define double @f6(<2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f6:
; CHECK: wfadb %f0, %v24, %v26
; CHECK: br %r14
  %scalar1 = extractelement <2 x double> %val1, i32 0
  %scalar2 = extractelement <2 x double> %val2, i32 0
  %ret = fadd double %scalar1, %scalar2
  ret double %ret
}
