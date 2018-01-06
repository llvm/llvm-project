; RUN: llc < %s -mtriple=aarch64-- | FileCheck %s

; Check that building up a vector w/ only one non-zero lane initializes
; intelligently.

define <8 x i8> @v8i8(i8 %t, i8 %s) nounwind {
  %v = insertelement <8 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 undef>, i8 %s, i32 7
  ret <8 x i8> %v

; CHECK: movi v[[R:[0-9]+]].8b, #0
; CHECK: mov  v[[R]].b[7], w{{[0-9]+}}
}

define <16 x i8> @v16i8(i8 %t, i8 %s) nounwind {
  %v = insertelement <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 undef>, i8 %s, i32 15
  ret <16 x i8> %v

; CHECK: movi v[[R:[0-9]+]].16b, #0
; CHECK: mov  v[[R]].b[15], w{{[0-9]+}}
}

define <4 x i16> @v4i16(i16 %t, i16 %s) nounwind {
  %v = insertelement <4 x i16> <i16 0, i16 0, i16 0, i16 undef>, i16 %s, i32 3
  ret <4 x i16> %v

; CHECK: movi v[[R:[0-9]+]].4h, #0
; CHECK: mov  v[[R]].h[3], w{{[0-9]+}}
}

define <8 x i16> @v8i16(i16 %t, i16 %s) nounwind {
  %v = insertelement <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 undef>, i16 %s, i32 7
  ret <8 x i16> %v

; CHECK: movi v[[R:[0-9]+]].8h, #0
; CHECK: mov  v[[R]].h[7], w{{[0-9]+}}
}

define <2 x i32> @v2i32(i32 %t, i32 %s) nounwind {
  %v = insertelement <2 x i32> <i32 0, i32 undef>, i32 %s, i32 1
  ret <2 x i32> %v

; CHECK: movi v[[R:[0-9]+]].2s, #0
; CHECK: mov  v[[R]].s[1], w{{[0-9]+}}
}

define <4 x i32> @v4i32(i32 %t, i32 %s) nounwind {
  %v = insertelement <4 x i32> <i32 0, i32 0, i32 0, i32 undef>, i32 %s, i32 3
  ret <4 x i32> %v

; CHECK: movi v[[R:[0-9]+]].4s, #0
; CHECK: mov  v[[R]].s[3], w{{[0-9]+}}
}

define <2 x i64> @v2i64(i64 %t, i64 %s) nounwind {
  %v = insertelement <2 x i64> <i64 0, i64 undef>, i64 %s, i32 1
  ret <2 x i64> %v

; CHECK: movi v[[R:[0-9]+]].2d, #0
; CHECK: mov  v[[R]].d[1], x{{[0-9]+}}
}

define <2 x float> @v2f32(float %t, float %s) nounwind {
  %v = insertelement <2 x float> <float 0.0, float undef>, float %s, i32 1
  ret <2 x float> %v

; CHECK: movi v[[R:[0-9]+]].2s, #0
; CHECK: mov  v[[R]].s[1], v{{[0-9]+}}.s[0]
}

define <4 x float> @v4f32(float %t, float %s) nounwind {
  %v = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float undef>, float %s, i32 3
  ret <4 x float> %v

; CHECK: movi v[[R:[0-9]+]].4s, #0
; CHECK: mov  v[[R]].s[3], v{{[0-9]+}}.s[0]
}

define <2 x double> @v2f64(double %t, double %s) nounwind {
  %v = insertelement <2 x double> <double 0.0, double undef>, double %s, i32 1
  ret <2 x double> %v

; CHECK: movi v[[R:[0-9]+]].2d, #0
; CHECK: mov  v[[R]].d[1], v{{[0-9]+}}.d[0]
}
