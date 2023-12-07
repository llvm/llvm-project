; Test vector maximum/minimum with a zero splat on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

define <2 x double> @f1(<2 x double> %val) {
; CHECK-LABEL: f1:
; CHECK: vgbm %v0, 0
; CHECK-NEXT: vfmaxdb %v24, %v24, %v0, 4
; CHECK-NEXT: br %r14
  %cmp = fcmp ogt <2 x double> %val,  zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x double> %val, <2 x double> zeroinitializer
  ret <2 x double> %ret
}

define <2 x double> @f2(<2 x double> %val) {
; CHECK-LABEL: f2:
; CHECK: vgbm %v0, 0
; CHECK-NEXT: vfmindb %v24, %v24, %v0, 4
; CHECK-NEXT: br %r14
  %cmp = fcmp olt <2 x double> %val,  zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x double> %val, <2 x double> zeroinitializer
  ret <2 x double> %ret
}

define <4 x float> @f3(<4 x float> %val) {
; CHECK-LABEL: f3:
; CHECK: vgbm %v0, 0
; CHECK-NEXT: vfmaxsb %v24, %v24, %v0, 4
; CHECK-NEXT: br %r14
  %cmp = fcmp ogt <4 x float> %val,  zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x float> %val, <4 x float> zeroinitializer
  ret <4 x float> %ret
}

define <4 x float> @f4(<4 x float> %val) {
; CHECK-LABEL: f4:
; CHECK: vgbm %v0, 0
; CHECK-NEXT: vfminsb %v24, %v24, %v0, 4
; CHECK-NEXT: br %r14
  %cmp = fcmp olt <4 x float> %val,  zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x float> %val, <4 x float> zeroinitializer
  ret <4 x float> %ret
}

define <2 x double> @f5(<2 x double> %val) {
; CHECK-LABEL: f5:
; CHECK: vgbm %v0, 0
; CHECK-NEXT: vfchedb	%v1, %v0, %v24
; CHECK-NEXT: vsel	%v24, %v0, %v24, %v1
; CHECK-NEXT: br %r14
  %cmp = fcmp ugt <2 x double> %val,  zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x double> %val, <2 x double> zeroinitializer
  ret <2 x double> %ret
}

define <2 x double> @f6(<2 x double> %val) {
; CHECK-LABEL: f6:
; CHECK: vgbm %v0, 0
; CHECK-NEXT: vfchedb	%v1, %v24, %v0
; CHECK-NEXT: vsel	%v24, %v0, %v24, %v1
; CHECK-NEXT: br %r14
  %cmp = fcmp ult <2 x double> %val,  zeroinitializer
  %ret = select <2 x i1> %cmp, <2 x double> %val, <2 x double> zeroinitializer
  ret <2 x double> %ret
}

define <4 x float> @f7(<4 x float> %val) {
; CHECK-LABEL: f7:
; CHECK: vgbm %v0, 0
; CHECK-NEXT: vfchesb	%v1, %v0, %v24
; CHECK-NEXT: vsel	%v24, %v0, %v24, %v1
; CHECK-NEXT: br %r14
  %cmp = fcmp ugt <4 x float> %val,  zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x float> %val, <4 x float> zeroinitializer
  ret <4 x float> %ret
}

define <4 x float> @f8(<4 x float> %val) {
; CHECK-LABEL: f8:
; CHECK: vgbm %v0, 0
; CHECK-NEXT: vfchesb	%v1, %v24, %v0
; CHECK-NEXT: vsel	%v24, %v0, %v24, %v1
; CHECK-NEXT: br %r14
  %cmp = fcmp ult <4 x float> %val,  zeroinitializer
  %ret = select <4 x i1> %cmp, <4 x float> %val, <4 x float> zeroinitializer
  ret <4 x float> %ret
}
