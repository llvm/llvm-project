; RUN: llc < %s -mtriple=aarch64 -mattr=neon -recip=!sqrt,!vec-sqrt | FileCheck %s --check-prefix=FAULT
; RUN: llc < %s -mtriple=aarch64 -mattr=neon -recip=sqrt,vec-sqrt   | FileCheck %s
; RUN: llc < %s -mtriple=aarch64 -mattr=neon,-use-reverse-square-root  | FileCheck %s --check-prefix=FAULT
; RUN: llc < %s -mtriple=aarch64 -mattr=neon,+use-reverse-square-root | FileCheck %s

declare float @llvm.sqrt.f32(float) #1
declare double @llvm.sqrt.f64(double) #1
declare <2 x float> @llvm.sqrt.v2f32(<2 x float>) #1
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>) #1
declare <2 x double> @llvm.sqrt.v2f64(<2 x double>) #1

define float @fsqrt(float %a) #0 {
  %1 = tail call fast float @llvm.sqrt.f32(float %a)
  ret float %1

; FAULT-LABEL: fsqrt:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: fsqrt:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frsqrte
}

define <2 x float> @f2sqrt(<2 x float> %a) #0 {
  %1 = tail call fast <2 x float> @llvm.sqrt.v2f32(<2 x float> %a) #2
  ret <2 x float> %1

; FAULT-LABEL: f2sqrt:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: f2sqrt:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: mov
; CHECK-NEXT: frsqrte
}

define <4 x float> @f4sqrt(<4 x float> %a) #0 {
  %1 = tail call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> %a) #2
  ret <4 x float> %1

; FAULT-LABEL: f4sqrt:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: f4sqrt:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: mov
; CHECK-NEXT: frsqrte
}

define double @dsqrt(double %a) #0 {
  %1 = tail call fast double @llvm.sqrt.f64(double %a)
  ret double %1

; FAULT-LABEL: dsqrt:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: dsqrt:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frsqrte
}

define <2 x double> @d2sqrt(<2 x double> %a) #0 {
  %1 = tail call fast <2 x double> @llvm.sqrt.v2f64(<2 x double> %a) #2
  ret <2 x double> %1

; FAULT-LABEL: d2sqrt:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: d2sqrt:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: mov
; CHECK-NEXT: frsqrte
}

define float @frsqrt(float %a) #0 {
  %1 = tail call fast float @llvm.sqrt.f32(float %a)
  %2 = fdiv fast float 1.000000e+00, %1
  ret float %2

; FAULT-LABEL: frsqrt:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: frsqrt:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frsqrte
}

define <2 x float> @f2rsqrt(<2 x float> %a) #0 {
  %1 = tail call fast <2 x float> @llvm.sqrt.v2f32(<2 x float> %a) #2
  %2 = fdiv fast <2 x float> <float 1.000000e+00, float 1.000000e+00>, %1
  ret <2 x float> %2

; FAULT-LABEL: f2rsqrt:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: f2rsqrt:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frsqrte
}

define <4 x float> @f4rsqrt(<4 x float> %a) #0 {
  %1 = tail call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> %a) #2
  %2 = fdiv fast <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %1
  ret <4 x float> %2

; FAULT-LABEL: f4rsqrt:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: f4rsqrt:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frsqrte
}

define double @drsqrt(double %a) #0 {
  %1 = tail call fast double @llvm.sqrt.f64(double %a)
  %2 = fdiv fast double 1.000000e+00, %1
  ret double %2

; FAULT-LABEL: drsqrt:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: drsqrt:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frsqrte
}

define <2 x double> @d2rsqrt(<2 x double> %a) #0 {
  %1 = tail call fast <2 x double> @llvm.sqrt.v2f64(<2 x double> %a) #2
  %2 = fdiv fast <2 x double> <double 1.000000e+00, double 1.000000e+00>, %1
  ret <2 x double> %2

; FAULT-LABEL: d2rsqrt:
; FAULT-NEXT: BB#0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: d2rsqrt:
; CHECK-NEXT: BB#0
; CHECK-NEXT: fmov
; CHECK-NEXT: frsqrte
}

attributes #0 = { nounwind "unsafe-fp-math"="true" }
