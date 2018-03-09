; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -mattr=+neon,-use-reciprocal-square-root | FileCheck %s --check-prefix=FAULT
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -mattr=+neon,+use-reciprocal-square-root | FileCheck %s

declare float @llvm.sqrt.f32(float) #0
declare <2 x float> @llvm.sqrt.v2f32(<2 x float>) #0
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>) #0
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #0
declare double @llvm.sqrt.f64(double) #0
declare <2 x double> @llvm.sqrt.v2f64(<2 x double>) #0
declare <4 x double> @llvm.sqrt.v4f64(<4 x double>) #0

define float @fsqrt(float %a) #0 {
  %1 = tail call fast float @llvm.sqrt.f32(float %a)
  ret float %1

; FAULT-LABEL: fsqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: fsqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:s[0-7]]]
; CHECK-NEXT: fmul [[RB:s[0-7]]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{s[0-7](, s[0-7])?}}, [[RB]]
; CHECK: frsqrts {{s[0-7]}}, {{s[0-7]}}, {{s[0-7]}}
; CHECK-NOT: frsqrts {{s[0-7]}}, {{s[0-7]}}, {{s[0-7]}}
; CHECK: fcmp {{s[0-7]}}, #0
}

define <2 x float> @f2sqrt(<2 x float> %a) #0 {
  %1 = tail call fast <2 x float> @llvm.sqrt.v2f32(<2 x float> %a)
  ret <2 x float> %1

; FAULT-LABEL: f2sqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: f2sqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:v[0-7]\.2s]]
; CHECK-NEXT: fmul [[RB:v[0-7]\.2s]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{v[0-7]\.2s(, v[0-7]\.2s)?}}, [[RB]]
; CHECK: frsqrts {{v[0-7]\.2s}}, {{v[0-7]\.2s}}, {{v[0-7]\.2s}}
; CHECK-NOT: frsqrts {{v[0-7]\.2s}}, {{v[0-7]\.2s}}, {{v[0-7]\.2s}}
; CHECK: fcmeq {{v[0-7]\.2s}}, {{v[0-7]\.2s}}, #0
}

define <4 x float> @f4sqrt(<4 x float> %a) #0 {
  %1 = tail call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> %a)
  ret <4 x float> %1

; FAULT-LABEL: f4sqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: f4sqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:v[0-7]\.4s]]
; CHECK-NEXT: fmul [[RB:v[0-7]\.4s]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{v[0-7]\.4s(, v[0-7]\.4s)?}}, [[RB]]
; CHECK: frsqrts {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK-NOT: frsqrts {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK: fcmeq {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, #0
}

define <8 x float> @f8sqrt(<8 x float> %a) #0 {
  %1 = tail call fast <8 x float> @llvm.sqrt.v8f32(<8 x float> %a)
  ret <8 x float> %1

; FAULT-LABEL: f8sqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt
; FAULT-NEXT: fsqrt

; CHECK-LABEL: f8sqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:v[0-7]\.4s]]
; CHECK-NEXT: fmul [[RB:v[0-7]\.4s]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{v[0-7]\.4s(, v[0-7]\.4s)?}}, [[RB]]
; CHECK: frsqrts {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK: fcmeq {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, #0
; CHECK: frsqrte [[RC:v[0-7]\.4s]]
; CHECK-NEXT: fmul [[RD:v[0-7]\.4s]], [[RC]], [[RC]]
; CHECK-NEXT: frsqrts {{v[0-7]\.4s(, v[0-7]\.4s)?}}, [[RD]]
; CHECK: frsqrts {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK-NOT: frsqrts {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK: fcmeq {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, #0
}

define double @dsqrt(double %a) #0 {
  %1 = tail call fast double @llvm.sqrt.f64(double %a)
  ret double %1

; FAULT-LABEL: dsqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: dsqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:d[0-7]]]
; CHECK-NEXT: fmul [[RB:d[0-7]]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{d[0-7](, d[0-7])?}}, [[RB]]
; CHECK: frsqrts {{d[0-7]}}, {{d[0-7]}}, {{d[0-7]}}
; CHECK: frsqrts {{d[0-7]}}, {{d[0-7]}}, {{d[0-7]}}
; CHECK-NOT: frsqrts {{d[0-7]}}, {{d[0-7]}}, {{d[0-7]}}
; CHECK: fcmp {{d[0-7]}}, #0
}

define <2 x double> @d2sqrt(<2 x double> %a) #0 {
  %1 = tail call fast <2 x double> @llvm.sqrt.v2f64(<2 x double> %a)
  ret <2 x double> %1

; FAULT-LABEL: d2sqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: d2sqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:v[0-7]\.2d]]
; CHECK-NEXT: fmul [[RB:v[0-7]\.2d]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{v[0-7]\.2d(, v[0-7]\.2d)?}}, [[RB]]
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK-NOT: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: fcmeq {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, #0
}

define <4 x double> @d4sqrt(<4 x double> %a) #0 {
  %1 = tail call fast <4 x double> @llvm.sqrt.v4f64(<4 x double> %a)
  ret <4 x double> %1

; FAULT-LABEL: d4sqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt
; FAULT-NEXT: fsqrt

; CHECK-LABEL: d4sqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:v[0-7]\.2d]]
; CHECK-NEXT: fmul [[RB:v[0-7]\.2d]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{v[0-7]\.2d(, v[0-7]\.2d)?}}, [[RB]]
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK-NOT: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: fcmeq {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, #0
; CHECK: frsqrte [[RC:v[0-7]\.2d]]
; CHECK-NEXT: fmul [[RD:v[0-7]\.2d]], [[RC]], [[RC]]
; CHECK-NEXT: frsqrts {{v[0-7]\.2d(, v[0-7]\.2d)?}}, [[RD]]
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK-NOT: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: fcmeq {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, #0
}

define float @frsqrt(float %a) #0 {
  %1 = tail call fast float @llvm.sqrt.f32(float %a)
  %2 = fdiv fast float 1.000000e+00, %1
  ret float %2

; FAULT-LABEL: frsqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: frsqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:s[0-7]]]
; CHECK-NEXT: fmul [[RB:s[0-7]]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{s[0-7](, s[0-7])?}}, [[RB]]
; CHECK: frsqrts {{s[0-7]}}, {{s[0-7]}}, {{s[0-7]}}
; CHECK-NOT: frsqrts {{s[0-7]}}, {{s[0-7]}}, {{s[0-7]}}
; CHECK-NOT: fcmp {{s[0-7]}}, #0
}

define <2 x float> @f2rsqrt(<2 x float> %a) #0 {
  %1 = tail call fast <2 x float> @llvm.sqrt.v2f32(<2 x float> %a)
  %2 = fdiv fast <2 x float> <float 1.000000e+00, float 1.000000e+00>, %1
  ret <2 x float> %2

; FAULT-LABEL: f2rsqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: f2rsqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:v[0-7]\.2s]]
; CHECK-NEXT: fmul [[RB:v[0-7]\.2s]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{v[0-7]\.2s(, v[0-7]\.2s)?}}, [[RB]]
; CHECK: frsqrts {{v[0-7]\.2s}}, {{v[0-7]\.2s}}, {{v[0-7]\.2s}}
; CHECK-NOT: frsqrts {{v[0-7]\.2s}}, {{v[0-7]\.2s}}, {{v[0-7]\.2s}}
; CHECK-NOT: fcmeq {{v[0-7]\.2s}}, {{v[0-7]\.2s}}, #0
}

define <4 x float> @f4rsqrt(<4 x float> %a) #0 {
  %1 = tail call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> %a)
  %2 = fdiv fast <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %1
  ret <4 x float> %2

; FAULT-LABEL: f4rsqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: f4rsqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:v[0-7]\.4s]]
; CHECK-NEXT: fmul [[RB:v[0-7]\.4s]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{v[0-7]\.4s(, v[0-7]\.4s)?}}, [[RB]]
; CHECK: frsqrts {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK-NOT: frsqrts {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK-NOT: fcmeq {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, #0
}

define <8 x float> @f8rsqrt(<8 x float> %a) #0 {
  %1 = tail call fast <8 x float> @llvm.sqrt.v8f32(<8 x float> %a)
  %2 = fdiv fast <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %1
  ret <8 x float> %2

; FAULT-LABEL: f8rsqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt
; FAULT-NEXT: fsqrt

; CHECK-LABEL: f8rsqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:v[0-7]\.4s]]
; CHECK: fmul [[RB:v[0-7]\.4s]], [[RA]], [[RA]]
; CHECK: frsqrts {{v[0-7]\.4s(, v[0-7]\.4s)?}}, [[RB]]
; CHECK: frsqrts {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK: frsqrts {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK: frsqrts {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK-NOT: frsqrts {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, {{v[0-7]\.4s}}
; CHECK-NOT: fcmeq {{v[0-7]\.4s}}, {{v[0-7]\.4s}}, #0
}

define double @drsqrt(double %a) #0 {
  %1 = tail call fast double @llvm.sqrt.f64(double %a)
  %2 = fdiv fast double 1.000000e+00, %1
  ret double %2

; FAULT-LABEL: drsqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: drsqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:d[0-7]]]
; CHECK-NEXT: fmul [[RB:d[0-7]]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{d[0-7](, d[0-7])?}}, [[RB]]
; CHECK: frsqrts {{d[0-7]}}, {{d[0-7]}}, {{d[0-7]}}
; CHECK: frsqrts {{d[0-7]}}, {{d[0-7]}}, {{d[0-7]}}
; CHECK-NOT: frsqrts {{d[0-7]}}, {{d[0-7]}}, {{d[0-7]}}
; CHECK-NOT: fcmp d0, #0
}

define <2 x double> @d2rsqrt(<2 x double> %a) #0 {
  %1 = tail call fast <2 x double> @llvm.sqrt.v2f64(<2 x double> %a)
  %2 = fdiv fast <2 x double> <double 1.000000e+00, double 1.000000e+00>, %1
  ret <2 x double> %2

; FAULT-LABEL: d2rsqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt

; CHECK-LABEL: d2rsqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:v[0-7]\.2d]]
; CHECK-NEXT: fmul [[RB:v[0-7]\.2d]], [[RA]], [[RA]]
; CHECK-NEXT: frsqrts {{v[0-7]\.2d(, v[0-7]\.2d)?}}, [[RB]]
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK-NOT: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK-NOT: fcmeq {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, #0
}

define <4 x double> @d4rsqrt(<4 x double> %a) #0 {
  %1 = tail call fast <4 x double> @llvm.sqrt.v4f64(<4 x double> %a)
  %2 = fdiv fast <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>, %1
  ret <4 x double> %2

; FAULT-LABEL: d4rsqrt:
; FAULT-NEXT: %bb.0
; FAULT-NEXT: fsqrt
; FAULT-NEXT: fsqrt

; CHECK-LABEL: d4rsqrt:
; CHECK-NEXT: %bb.0
; CHECK-NEXT: frsqrte [[RA:v[0-7]\.2d]]
; CHECK: fmul [[RB:v[0-7]\.2d]], [[RA]], [[RA]]
; CHECK: frsqrts {{v[0-7]\.2d(, v[0-7]\.2d)?}}, [[RB]]
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK-NOT: frsqrts {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, {{v[0-7]\.2d}}
; CHECK-NOT: fcmeq {{v[0-7]\.2d}}, {{v[0-7]\.2d}}, #0
}

attributes #0 = { nounwind "unsafe-fp-math"="true" }
