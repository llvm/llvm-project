; RUN: llc < %s -mtriple=aarch64 -mattr=neon -recip=!div,!vec-div | FileCheck %s --check-prefix=FAULT
; RUN: llc < %s -mtriple=aarch64 -mattr=neon -recip=div,vec-div   | FileCheck %s

define float @frecp(float %x) #0 {
  %div = fdiv fast float 1.0, %x
  ret float %div

; FAULT-LABEL: frecp:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fmov
; FAULT-NEXT: fdiv

; CHECK-LABEL: frecp:
; CHECK-NEXT: BB#0
; CHECK-NEXT: frecpe
; CHECK-NEXT: fmov
}

define <2 x float> @f2recp(<2 x float> %x) #0 {
  %div = fdiv fast <2 x float> <float 1.0, float 1.0>, %x
  ret <2 x float> %div

; FAULT-LABEL: f2recp:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fmov
; FAULT-NEXT: fdiv

; CHECK-LABEL: f2recp:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frecpe
}

define <4 x float> @f4recp(<4 x float> %x) #0 {
  %div = fdiv fast <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, %x
  ret <4 x float> %div

; FAULT-LABEL: f4recp:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fmov
; FAULT-NEXT: fdiv

; CHECK-LABEL: f4recp:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frecpe
}

define <8 x float> @f8recp(<8 x float> %x) #0 {
  %div = fdiv fast <8 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>, %x
  ret <8 x float> %div

; FAULT-LABEL: f8recp:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fmov
; FAULT-NEXT: fdiv
; FAULT-NEXT: fdiv

; CHECK-LABEL: f8recp:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frecpe
; CHECK: frecpe
}

define double @drecp(double %x) #0 {
  %div = fdiv fast double 1.0, %x
  ret double %div

; FAULT-LABEL: drecp:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fmov
; FAULT-NEXT: fdiv

; CHECK-LABEL: drecp:
; CHECK-NEXT: BB#0
; CHECK-NEXT: frecpe
; CHECK-NEXT: fmov
}

define <2 x double> @d2recp(<2 x double> %x) #0 {
  %div = fdiv fast <2 x double> <double 1.0, double 1.0>, %x
  ret <2 x double> %div

; FAULT-LABEL: d2recp:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fmov
; FAULT-NEXT: fdiv

; CHECK-LABEL: d2recp:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frecpe
}

define <4 x double> @d4recp(<4 x double> %x) #0 {
  %div = fdiv fast <4 x double> <double 1.0, double 1.0, double 1.0, double 1.0>, %x
  ret <4 x double> %div

; FAULT-LABEL: d4recp:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fmov
; FAULT-NEXT: fdiv
; FAULT-NEXT: fdiv

; CHECK-LABEL: d4recp:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frecpe
; CHECK: frecpe
}

attributes #0 = { nounwind "unsafe-fp-math"="true" }
