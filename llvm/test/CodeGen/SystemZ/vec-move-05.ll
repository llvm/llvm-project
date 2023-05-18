; Test vector extraction.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 extraction of the first element.
define i8 @f1(<16 x i8> %val) {
; CHECK-LABEL: f1:
; CHECK: vlgvb %r2, %v24, 0
; CHECK: br %r14
  %ret = extractelement <16 x i8> %val, i32 0
  ret i8 %ret
}

; Test v16i8 extraction of the last element.
define i8 @f2(<16 x i8> %val) {
; CHECK-LABEL: f2:
; CHECK: vlgvb %r2, %v24, 15
; CHECK: br %r14
  %ret = extractelement <16 x i8> %val, i32 15
  ret i8 %ret
}

; Test v16i8 extractions of an absurd element number.  This must compile
; but we don't care what it does.
define i8 @f3(<16 x i8> %val) {
; CHECK-LABEL: f3:
; CHECK-NOT: vlgvb %r2, %v24, 100000
; CHECK: br %r14
  %ret = extractelement <16 x i8> %val, i32 100000
  ret i8 %ret
}

; Test v16i8 extraction of a variable element.
define i8 @f4(<16 x i8> %val, i32 %index) {
; CHECK-LABEL: f4:
; CHECK: vlgvb %r2, %v24, 0(%r2)
; CHECK: br %r14
  %ret = extractelement <16 x i8> %val, i32 %index
  ret i8 %ret
}

; Test v8i16 extraction of the first element.
define i16 @f5(<8 x i16> %val) {
; CHECK-LABEL: f5:
; CHECK: vlgvh %r2, %v24, 0
; CHECK: br %r14
  %ret = extractelement <8 x i16> %val, i32 0
  ret i16 %ret
}

; Test v8i16 extraction of the last element.
define i16 @f6(<8 x i16> %val) {
; CHECK-LABEL: f6:
; CHECK: vlgvh %r2, %v24, 7
; CHECK: br %r14
  %ret = extractelement <8 x i16> %val, i32 7
  ret i16 %ret
}

; Test v8i16 extractions of an absurd element number.  This must compile
; but we don't care what it does.
define i16 @f7(<8 x i16> %val) {
; CHECK-LABEL: f7:
; CHECK-NOT: vlgvh %r2, %v24, 100000
; CHECK: br %r14
  %ret = extractelement <8 x i16> %val, i32 100000
  ret i16 %ret
}

; Test v8i16 extraction of a variable element.
define i16 @f8(<8 x i16> %val, i32 %index) {
; CHECK-LABEL: f8:
; CHECK: vlgvh %r2, %v24, 0(%r2)
; CHECK: br %r14
  %ret = extractelement <8 x i16> %val, i32 %index
  ret i16 %ret
}

; Test v4i32 extraction of the first element.
define i32 @f9(<4 x i32> %val) {
; CHECK-LABEL: f9:
; CHECK: vlgvf %r2, %v24, 0
; CHECK: br %r14
  %ret = extractelement <4 x i32> %val, i32 0
  ret i32 %ret
}

; Test v4i32 extraction of the last element.
define i32 @f10(<4 x i32> %val) {
; CHECK-LABEL: f10:
; CHECK: vlgvf %r2, %v24, 3
; CHECK: br %r14
  %ret = extractelement <4 x i32> %val, i32 3
  ret i32 %ret
}

; Test v4i32 extractions of an absurd element number.  This must compile
; but we don't care what it does.
define i32 @f11(<4 x i32> %val) {
; CHECK-LABEL: f11:
; CHECK-NOT: vlgvf %r2, %v24, 100000
; CHECK: br %r14
  %ret = extractelement <4 x i32> %val, i32 100000
  ret i32 %ret
}

; Test v4i32 extraction of a variable element.
define i32 @f12(<4 x i32> %val, i32 %index) {
; CHECK-LABEL: f12:
; CHECK: vlgvf %r2, %v24, 0(%r2)
; CHECK: br %r14
  %ret = extractelement <4 x i32> %val, i32 %index
  ret i32 %ret
}

; Test v2i64 extraction of the first element.
define i64 @f13(<2 x i64> %val) {
; CHECK-LABEL: f13:
; CHECK: vlgvg %r2, %v24, 0
; CHECK: br %r14
  %ret = extractelement <2 x i64> %val, i32 0
  ret i64 %ret
}

; Test v2i64 extraction of the last element.
define i64 @f14(<2 x i64> %val) {
; CHECK-LABEL: f14:
; CHECK: vlgvg %r2, %v24, 1
; CHECK: br %r14
  %ret = extractelement <2 x i64> %val, i32 1
  ret i64 %ret
}

; Test v2i64 extractions of an absurd element number.  This must compile
; but we don't care what it does.
define i64 @f15(<2 x i64> %val) {
; CHECK-LABEL: f15:
; CHECK-NOT: vlgvg %r2, %v24, 100000
; CHECK: br %r14
  %ret = extractelement <2 x i64> %val, i32 100000
  ret i64 %ret
}

; Test v2i64 extraction of a variable element.
define i64 @f16(<2 x i64> %val, i32 %index) {
; CHECK-LABEL: f16:
; CHECK: vlgvg %r2, %v24, 0(%r2)
; CHECK: br %r14
  %ret = extractelement <2 x i64> %val, i32 %index
  ret i64 %ret
}

; Test v4f32 extraction of element 0.
define float @f17(<4 x float> %val) {
; CHECK-LABEL: f17:
; CHECK: vlr %v0, %v24
; CHECK: br %r14
  %ret = extractelement <4 x float> %val, i32 0
  ret float %ret
}

; Test v4f32 extraction of element 1.
define float @f18(<4 x float> %val) {
; CHECK-LABEL: f18:
; CHECK: vrepf %v0, %v24, 1
; CHECK: br %r14
  %ret = extractelement <4 x float> %val, i32 1
  ret float %ret
}

; Test v4f32 extraction of element 2.
define float @f19(<4 x float> %val) {
; CHECK-LABEL: f19:
; CHECK: vrepf %v0, %v24, 2
; CHECK: br %r14
  %ret = extractelement <4 x float> %val, i32 2
  ret float %ret
}

; Test v4f32 extraction of element 3.
define float @f20(<4 x float> %val) {
; CHECK-LABEL: f20:
; CHECK: vrepf %v0, %v24, 3
; CHECK: br %r14
  %ret = extractelement <4 x float> %val, i32 3
  ret float %ret
}

; Test v4f32 extractions of an absurd element number.  This must compile
; but we don't care what it does.
define float @f21(<4 x float> %val) {
  %ret = extractelement <4 x float> %val, i32 100000
  ret float %ret
}

; Test v4f32 extraction of a variable element.
define float @f22(<4 x float> %val, i32 %index) {
; CHECK-LABEL: f22:
; CHECK: vlgvf [[REG:%r[0-5]]], %v24, 0(%r2)
; CHECK: vlvgf %v0, [[REG]], 0
; CHECK: br %r14
  %ret = extractelement <4 x float> %val, i32 %index
  ret float %ret
}

; Test v2f64 extraction of the first element.
define double @f23(<2 x double> %val) {
; CHECK-LABEL: f23:
; CHECK: vlr %v0, %v24
; CHECK: br %r14
  %ret = extractelement <2 x double> %val, i32 0
  ret double %ret
}

; Test v2f64 extraction of the last element.
define double @f24(<2 x double> %val) {
; CHECK-LABEL: f24:
; CHECK: vrepg %v0, %v24, 1
; CHECK: br %r14
  %ret = extractelement <2 x double> %val, i32 1
  ret double %ret
}

; Test v2f64 extractions of an absurd element number.  This must compile
; but we don't care what it does.
define double @f25(<2 x double> %val) {
  %ret = extractelement <2 x double> %val, i32 100000
  ret double %ret
}

; Test v2f64 extraction of a variable element.
define double @f26(<2 x double> %val, i32 %index) {
; CHECK-LABEL: f26:
; CHECK: vlgvg [[REG:%r[0-5]]], %v24, 0(%r2)
; CHECK: ldgr %f0, [[REG]]
; CHECK: br %r14
  %ret = extractelement <2 x double> %val, i32 %index
  ret double %ret
}

; Test v16i8 extraction of a variable element with an offset.
define i8 @f27(<16 x i8> %val, i32 %index) {
; CHECK-LABEL: f27:
; CHECK: vlgvb %r2, %v24, 1(%r2)
; CHECK: br %r14
  %add = add i32 %index, 1
  %ret = extractelement <16 x i8> %val, i32 %add
  ret i8 %ret
}
