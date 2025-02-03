; Test vector division on arch15.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=arch15 | FileCheck %s

; Test a v4i32 signed division.
define <4 x i32> @f1(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f1:
; CHECK: vdf %v24, %v26, %v28
; CHECK: br %r14
  %ret = sdiv <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v4i32 unsigned division.
define <4 x i32> @f2(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f2:
; CHECK: vdlf %v24, %v26, %v28
; CHECK: br %r14
  %ret = udiv <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v4i32 signed remainder.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vrf %v24, %v26, %v28
; CHECK: br %r14
  %ret = srem <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v4i32 unsigned remainder.
define <4 x i32> @f4(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f4:
; CHECK: vrlf %v24, %v26, %v28
; CHECK: br %r14
  %ret = urem <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v2i64 signed division.
define <2 x i64> @f5(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f5:
; CHECK: vdg %v24, %v26, %v28
; CHECK: br %r14
  %ret = sdiv <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}

; Test a v2i64 unsigned division.
define <2 x i64> @f6(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f6:
; CHECK: vdlg %v24, %v26, %v28
; CHECK: br %r14
  %ret = udiv <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}

; Test a v2i64 signed remainder.
define <2 x i64> @f7(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f7:
; CHECK: vrg %v24, %v26, %v28
; CHECK: br %r14
  %ret = srem <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}

; Test a v2i64 unsigned remainder.
define <2 x i64> @f8(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f8:
; CHECK: vrlg %v24, %v26, %v28
; CHECK: br %r14
  %ret = urem <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}

