; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -mattr=+neon | FileCheck %s

define float @frecp0(float %x) #0 {
  %div = fdiv fast float 1.0, %x
  ret float %div

; CHECK-LABEL: frecp0:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: fmov
; CHECK-NEXT: fdiv
}

define float @frecp1(float %x) #1 {
  %div = fdiv fast float 1.0, %x
  ret float %div

; CHECK-LABEL: frecp1:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frecpe [[R:s[0-7]]]
; CHECK-NEXT: frecps {{s[0-7](, s[0-7])?}}, [[R]]
; CHECK: frecps {{s[0-7]}}, {{s[0-7]}}, {{s[0-7]}}
; CHECK-NOT: frecps {{s[0-7]}}, {{s[0-7]}}, {{s[0-7]}}
}

define <2 x float> @f2recp0(<2 x float> %x) #0 {
  %div = fdiv fast <2 x float> <float 1.0, float 1.0>, %x
  ret <2 x float> %div

; CHECK-LABEL: f2recp0:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: fmov
; CHECK-NEXT: fdiv
}

define <2 x float> @f2recp1(<2 x float> %x) #1 {
  %div = fdiv fast <2 x float> <float 1.0, float 1.0>, %x
  ret <2 x float> %div

; CHECK-LABEL: f2recp1:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frecpe [[R:v[0-7]\.2s]]
; CHECK-NEXT: frecps {{v[0-7]\.2s(, v[0-7].2s)?}}, [[R]]
; CHECK: frecps {{v[0-7]\.2s}}, {{v[0-7]\.2s}}, {{v[0-7]\.2s}}
; CHECK-NOT: frecps {{v[0-7]\.2s}}, {{v[0-7]\.2s}}, {{v[0-7]\.2s}}
}

define <4 x float> @f4recp0(<4 x float> %x) #0 {
  %div = fdiv fast <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, %x
  ret <4 x float> %div

; CHECK-LABEL: f4recp0:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: fmov
; CHECK-NEXT: fdiv
}

define <4 x float> @f4recp1(<4 x float> %x) #1 {
  %div = fdiv fast <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, %x
  ret <4 x float> %div

; CHECK-LABEL: f4recp1:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frecpe [[R:v[0-7]\.4s]]
; CHECK-NEXT: frecps {{v[0-7]\.4s(, v[0-7].4s)?}}, [[R]]
; CHECK: frecps {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK-NOT: frecps {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
}

define <8 x float> @f8recp0(<8 x float> %x) #0 {
  %div = fdiv fast <8 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>, %x
  ret <8 x float> %div

; CHECK-LABEL: f8recp0:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: fmov
; CHECK-NEXT: fdiv
; CHECK-NEXT: fdiv
}

define <8 x float> @f8recp1(<8 x float> %x) #1 {
  %div = fdiv fast <8 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>, %x
  ret <8 x float> %div

; CHECK-LABEL: f8recp1:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frecpe [[R:v[0-7]\.4s]]
; CHECK: frecps {{v[0-7]\.4s(, v[0-7].4s)?}}, [[R]]
; CHECK: frecps {{v[0-7]\.4s(, v[0-7].4s)?}}, {{v[0-7]\.4s}}
; CHECK: frecps {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK: frecps {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK-NOT: frecps {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
}

define double @drecp0(double %x) #0 {
  %div = fdiv fast double 1.0, %x
  ret double %div

; CHECK-LABEL: drecp0:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: fmov
; CHECK-NEXT: fdiv
}

define double @drecp1(double %x) #1 {
  %div = fdiv fast double 1.0, %x
  ret double %div

; CHECK-LABEL: drecp1:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frecpe [[R:d[0-7]]]
; CHECK-NEXT: frecps {{d[0-7](, d[0-7])?}}, [[R]]
; CHECK: frecps {{d[0-7]}}, {{d[0-7]}}, {{d[0-7]}}
; CHECK: frecps {{d[0-7]}}, {{d[0-7]}}, {{d[0-7]}}
; CHECK-NOT: frecps {{d[0-7]}}, {{d[0-7]}}, {{d[0-7]}}
}

define <2 x double> @d2recp0(<2 x double> %x) #0 {
  %div = fdiv fast <2 x double> <double 1.0, double 1.0>, %x
  ret <2 x double> %div

; CHECK-LABEL: d2recp0:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: fmov
; CHECK-NEXT: fdiv
}

define <2 x double> @d2recp1(<2 x double> %x) #1 {
  %div = fdiv fast <2 x double> <double 1.0, double 1.0>, %x
  ret <2 x double> %div

; CHECK-LABEL: d2recp1:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frecpe [[R:v[0-7]\.2d]]
; CHECK-NEXT: frecps {{v[0-7]\.2d(, v[0-7].2d)?}}, [[R]]
; CHECK: frecps {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frecps {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK-NOT: frecps {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
}

define <4 x double> @d4recp0(<4 x double> %x) #0 {
  %div = fdiv fast <4 x double> <double 1.0, double 1.0, double 1.0, double 1.0>, %x
  ret <4 x double> %div

; CHECK-LABEL: d4recp0:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: fmov
; CHECK-NEXT: fdiv
; CHECK-NEXT: fdiv
}

define <4 x double> @d4recp1(<4 x double> %x) #1 {
  %div = fdiv fast <4 x double> <double 1.0, double 1.0, double 1.0, double 1.0>, %x
  ret <4 x double> %div

; CHECK-LABEL: d4recp1:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frecpe [[R:v[0-7]\.2d]]
; CHECK: frecps {{v[0-7]\.2d(, v[0-7].2d)?}}, [[R]]
; CHECK: frecps {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frecps {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frecps {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frecps {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frecps {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK-NOT: frecps {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
}

attributes #0 = { nounwind "unsafe-fp-math"="true" }
attributes #1 = { nounwind "unsafe-fp-math"="true" "reciprocal-estimates"="div,vec-div" }
