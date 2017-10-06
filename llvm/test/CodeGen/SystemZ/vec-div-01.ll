; Test vector division.  There is no native integer support for this,
; so the integer cases are really a test of the operation legalization code.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 division.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vlvgp [[REG:%v[0-9]+]],
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 0
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 1
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 2
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 3
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 4
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 5
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 6
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 8
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 9
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 10
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 11
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 12
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 13
; CHECK-DAG: vlvgb [[REG]], {{%r[0-9]+}}, 14
; CHECK: br %r14
  %ret = sdiv <16 x i8> %val1, %val2
  ret <16 x i8> %ret
}

; Test a v8i16 division.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f2:
; CHECK: vlvgp [[REG:%v[0-9]+]],
; CHECK-DAG: vlvgh [[REG]], {{%r[0-9]+}}, 0
; CHECK-DAG: vlvgh [[REG]], {{%r[0-9]+}}, 1
; CHECK-DAG: vlvgh [[REG]], {{%r[0-9]+}}, 2
; CHECK-DAG: vlvgh [[REG]], {{%r[0-9]+}}, 4
; CHECK-DAG: vlvgh [[REG]], {{%r[0-9]+}}, 5
; CHECK-DAG: vlvgh [[REG]], {{%r[0-9]+}}, 6
; CHECK: br %r14
  %ret = sdiv <8 x i16> %val1, %val2
  ret <8 x i16> %ret
}

; Test a v4i32 division.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vlvgp [[REG:%v[0-9]+]],
; CHECK-DAG: vlvgf [[REG]], {{%r[0-9]+}}, 0
; CHECK-DAG: vlvgf [[REG]], {{%r[0-9]+}}, 2
; CHECK: br %r14
  %ret = sdiv <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v2i64 division.
define <2 x i64> @f4(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f4:
; CHECK: vlvgp %v24,
; CHECK: br %r14
  %ret = sdiv <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}

; Test a v2f64 division.
define <2 x double> @f5(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) {
; CHECK-LABEL: f5:
; CHECK: vfddb %v24, %v26, %v28
; CHECK: br %r14
  %ret = fdiv <2 x double> %val1, %val2
  ret <2 x double> %ret
}

; Test an f64 division that uses vector registers.
define double @f6(<2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f6:
; CHECK: wfddb %f0, %v24, %v26
; CHECK: br %r14
  %scalar1 = extractelement <2 x double> %val1, i32 0
  %scalar2 = extractelement <2 x double> %val2, i32 0
  %ret = fdiv double %scalar1, %scalar2
  ret double %ret
}
