; Test conversions between integer and float elements on z15.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

; Test conversion of f32s to signed i32s.
define <4 x i32> @f1(<4 x float> %floats) {
; CHECK-LABEL: f1:
; CHECK: vcfeb %v24, %v24, 0, 5
; CHECK: br %r14
  %dwords = fptosi <4 x float> %floats to <4 x i32>
  ret <4 x i32> %dwords
}

; Test conversion of f32s to unsigned i32s.
define <4 x i32> @f2(<4 x float> %floats) {
; CHECK-LABEL: f2:
; CHECK: vclfeb %v24, %v24, 0, 5
; CHECK: br %r14
  %dwords = fptoui <4 x float> %floats to <4 x i32>
  ret <4 x i32> %dwords
}

; Test conversion of signed i32s to f32s.
define <4 x float> @f3(<4 x i32> %dwords) {
; CHECK-LABEL: f3:
; CHECK: vcefb %v24, %v24, 0, 0
; CHECK: br %r14
  %floats = sitofp <4 x i32> %dwords to <4 x float>
  ret <4 x float> %floats
}

; Test conversion of unsigned i32s to f32s.
define <4 x float> @f4(<4 x i32> %dwords) {
; CHECK-LABEL: f4:
; CHECK: vcelfb %v24, %v24, 0, 0
; CHECK: br %r14
  %floats = uitofp <4 x i32> %dwords to <4 x float>
  ret <4 x float> %floats
}

