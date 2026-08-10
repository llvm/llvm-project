; Test conversions between integer and float elements.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test conversion of f64s to signed i64s.
define <2 x i64> @f1(<2 x double> %doubles) {
; CHECK-LABEL: f1:
; CHECK: vcgdb %v24, %v24, 0, 5
; CHECK: br %r14
  %dwords = fptosi <2 x double> %doubles to <2 x i64>
  ret <2 x i64> %dwords
}

; Test conversion of f64s to unsigned i64s.
define <2 x i64> @f2(<2 x double> %doubles) {
; CHECK-LABEL: f2:
; CHECK: vclgdb %v24, %v24, 0, 5
; CHECK: br %r14
  %dwords = fptoui <2 x double> %doubles to <2 x i64>
  ret <2 x i64> %dwords
}

; Test conversion of signed i64s to f64s.
define <2 x double> @f3(<2 x i64> %dwords) {
; CHECK-LABEL: f3:
; CHECK: vcdgb %v24, %v24, 0, 0
; CHECK: br %r14
  %doubles = sitofp <2 x i64> %dwords to <2 x double>
  ret <2 x double> %doubles
}

; Test conversion of unsigned i64s to f64s.
define <2 x double> @f4(<2 x i64> %dwords) {
; CHECK-LABEL: f4:
; CHECK: vcdlgb %v24, %v24, 0, 0
; CHECK: br %r14
  %doubles = uitofp <2 x i64> %dwords to <2 x double>
  ret <2 x double> %doubles
}

; Test conversion of f64s to signed i32s, which must compile.
define void @f5(<2 x double> %doubles, ptr %ptr) {
  %words = fptosi <2 x double> %doubles to <2 x i32>
  store <2 x i32> %words, ptr %ptr
  ret void
}

; Test conversion of f64s to unsigned i32s, which must compile.
define void @f6(<2 x double> %doubles, ptr %ptr) {
  %words = fptoui <2 x double> %doubles to <2 x i32>
  store <2 x i32> %words, ptr %ptr
  ret void
}

; Test conversion of signed i32s to f64s, which must compile.
define <2 x double> @f7(ptr %ptr) {
  %words = load <2 x i32>, ptr %ptr
  %doubles = sitofp <2 x i32> %words to <2 x double>
  ret <2 x double> %doubles
}

; Test conversion of unsigned i32s to f64s, which must compile.
define <2 x double> @f8(ptr %ptr) {
  %words = load <2 x i32>, ptr %ptr
  %doubles = uitofp <2 x i32> %words to <2 x double>
  ret <2 x double> %doubles
}

; Test conversion of f32s to signed i64s, which must compile.
define <2 x i64> @f9(ptr %ptr) {
  %floats = load <2 x float>, ptr %ptr
  %dwords = fptosi <2 x float> %floats to <2 x i64>
  ret <2 x i64> %dwords
}

; Test conversion of f32s to unsigned i64s, which must compile.
define <2 x i64> @f10(ptr %ptr) {
  %floats = load <2 x float>, ptr %ptr
  %dwords = fptoui <2 x float> %floats to <2 x i64>
  ret <2 x i64> %dwords
}

; Test conversion of signed i64s to f32, which must compile.
define void @f11(<2 x i64> %dwords, ptr %ptr) {
  %floats = sitofp <2 x i64> %dwords to <2 x float>
  store <2 x float> %floats, ptr %ptr
  ret void
}

; Test conversion of unsigned i64s to f32, which must compile.
define void @f12(<2 x i64> %dwords, ptr %ptr) {
  %floats = uitofp <2 x i64> %dwords to <2 x float>
  store <2 x float> %floats, ptr %ptr
  ret void
}
