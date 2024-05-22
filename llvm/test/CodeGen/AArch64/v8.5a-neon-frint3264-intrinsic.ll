; RUN: llc < %s -mtriple=aarch64-eabi -mattr=+v8.5a  | FileCheck %s

declare <2 x float> @llvm.aarch64.neon.frint32x.v2f32(<2 x float>)
declare <4 x float> @llvm.aarch64.neon.frint32x.v4f32(<4 x float>)
declare <2 x float> @llvm.aarch64.neon.frint32z.v2f32(<2 x float>)
declare <4 x float> @llvm.aarch64.neon.frint32z.v4f32(<4 x float>)

define dso_local <2 x float> @t_vrnd32x_f32(<2 x float> %a) {
; CHECK-LABEL: t_vrnd32x_f32:
; CHECK:         frint32x v0.2s, v0.2s
; CHECK-NEXT:    ret
entry:
  %val = tail call <2 x float> @llvm.aarch64.neon.frint32x.v2f32(<2 x float> %a)
  ret <2 x float> %val
}

define dso_local <4 x float> @t_vrnd32xq_f32(<4 x float> %a) {
; CHECK-LABEL: t_vrnd32xq_f32:
; CHECK:         frint32x v0.4s, v0.4s
; CHECK-NEXT:    ret
entry:
  %val = tail call <4 x float> @llvm.aarch64.neon.frint32x.v4f32(<4 x float> %a)
  ret <4 x float> %val
}

define dso_local <2 x float> @t_vrnd32z_f32(<2 x float> %a) {
; CHECK-LABEL: t_vrnd32z_f32:
; CHECK:         frint32z v0.2s, v0.2s
; CHECK-NEXT:    ret
entry:
  %val = tail call <2 x float> @llvm.aarch64.neon.frint32z.v2f32(<2 x float> %a)
  ret <2 x float> %val
}

define dso_local <4 x float> @t_vrnd32zq_f32(<4 x float> %a) {
; CHECK-LABEL: t_vrnd32zq_f32:
; CHECK:         frint32z v0.4s, v0.4s
; CHECK-NEXT:    ret
entry:
  %val = tail call <4 x float> @llvm.aarch64.neon.frint32z.v4f32(<4 x float> %a)
  ret <4 x float> %val
}

declare <2 x float> @llvm.aarch64.neon.frint64x.v2f32(<2 x float>)
declare <4 x float> @llvm.aarch64.neon.frint64x.v4f32(<4 x float>)
declare <2 x float> @llvm.aarch64.neon.frint64z.v2f32(<2 x float>)
declare <4 x float> @llvm.aarch64.neon.frint64z.v4f32(<4 x float>)

define dso_local <2 x float> @t_vrnd64x_f32(<2 x float> %a) {
; CHECK-LABEL: t_vrnd64x_f32:
; CHECK:         frint64x v0.2s, v0.2s
; CHECK-NEXT:    ret
entry:
  %val = tail call <2 x float> @llvm.aarch64.neon.frint64x.v2f32(<2 x float> %a)
  ret <2 x float> %val
}

define dso_local <4 x float> @t_vrnd64xq_f32(<4 x float> %a) {
; CHECK-LABEL: t_vrnd64xq_f32:
; CHECK:         frint64x v0.4s, v0.4s
; CHECK-NEXT:    ret
entry:
  %val = tail call <4 x float> @llvm.aarch64.neon.frint64x.v4f32(<4 x float> %a)
  ret <4 x float> %val
}

define dso_local <2 x float> @t_vrnd64z_f32(<2 x float> %a) {
; CHECK-LABEL: t_vrnd64z_f32:
; CHECK:         frint64z v0.2s, v0.2s
; CHECK-NEXT:    ret
entry:
  %val = tail call <2 x float> @llvm.aarch64.neon.frint64z.v2f32(<2 x float> %a)
  ret <2 x float> %val
}

define dso_local <4 x float> @t_vrnd64zq_f32(<4 x float> %a) {
; CHECK-LABEL: t_vrnd64zq_f32:
; CHECK:         frint64z v0.4s, v0.4s
; CHECK-NEXT:    ret
entry:
  %val = tail call <4 x float> @llvm.aarch64.neon.frint64z.v4f32(<4 x float> %a)
  ret <4 x float> %val
}

declare <1 x double> @llvm.aarch64.neon.frint32x.v1f64(<1 x double>)
declare <2 x double> @llvm.aarch64.neon.frint32x.v2f64(<2 x double>)
declare <1 x double> @llvm.aarch64.neon.frint32z.v1f64(<1 x double>)
declare <2 x double> @llvm.aarch64.neon.frint32z.v2f64(<2 x double>)

define dso_local <1 x double> @t_vrnd32x_f64(<1 x double> %a) {
; CHECK-LABEL: t_vrnd32x_f64:
; CHECK:         frint32x d0, d0
; CHECK-NEXT:    ret
entry:
  %val = tail call <1 x double> @llvm.aarch64.neon.frint32x.v1f64(<1 x double> %a)
  ret <1 x double> %val
}

define dso_local <2 x double> @t_vrnd32xq_f64(<2 x double> %a) {
; CHECK-LABEL: t_vrnd32xq_f64:
; CHECK:         frint32x v0.2d, v0.2d
; CHECK-NEXT:    ret
entry:
  %val = tail call <2 x double> @llvm.aarch64.neon.frint32x.v2f64(<2 x double> %a)
  ret <2 x double> %val
}

define dso_local <1 x double> @t_vrnd32z_f64(<1 x double> %a) {
; CHECK-LABEL: t_vrnd32z_f64:
; CHECK:         frint32z d0, d0
; CHECK-NEXT:    ret
entry:
  %val = tail call <1 x double> @llvm.aarch64.neon.frint32z.v1f64(<1 x double> %a)
  ret <1 x double> %val
}

define dso_local <2 x double> @t_vrnd32zq_f64(<2 x double> %a) {
; CHECK-LABEL: t_vrnd32zq_f64:
; CHECK:         frint32z v0.2d, v0.2d
; CHECK-NEXT:    ret
entry:
  %val = tail call <2 x double> @llvm.aarch64.neon.frint32z.v2f64(<2 x double> %a)
  ret <2 x double> %val
}

declare <1 x double> @llvm.aarch64.neon.frint64x.v1f64(<1 x double>)
declare <2 x double> @llvm.aarch64.neon.frint64x.v2f64(<2 x double>)
declare <1 x double> @llvm.aarch64.neon.frint64z.v1f64(<1 x double>)
declare <2 x double> @llvm.aarch64.neon.frint64z.v2f64(<2 x double>)

define dso_local <1 x double> @t_vrnd64x_f64(<1 x double> %a) {
; CHECK-LABEL: t_vrnd64x_f64:
; CHECK:         frint64x d0, d0
; CHECK-NEXT:    ret
entry:
  %val = tail call <1 x double> @llvm.aarch64.neon.frint64x.v1f64(<1 x double> %a)
  ret <1 x double> %val
}

define dso_local <2 x double> @t_vrnd64xq_f64(<2 x double> %a) {
; CHECK-LABEL: t_vrnd64xq_f64:
; CHECK:         frint64x v0.2d, v0.2d
; CHECK-NEXT:    ret
entry:
  %val = tail call <2 x double> @llvm.aarch64.neon.frint64x.v2f64(<2 x double> %a)
  ret <2 x double> %val
}

define dso_local <1 x double> @t_vrnd64z_f64(<1 x double> %a) {
; CHECK-LABEL: t_vrnd64z_f64:
; CHECK:         frint64z d0, d0
; CHECK-NEXT:    ret
entry:
  %val = tail call <1 x double> @llvm.aarch64.neon.frint64z.v1f64(<1 x double> %a)
  ret <1 x double> %val
}

define dso_local <2 x double> @t_vrnd64zq_f64(<2 x double> %a) {
; CHECK-LABEL: t_vrnd64zq_f64:
; CHECK:         frint64z v0.2d, v0.2d
; CHECK-NEXT:    ret
entry:
  %val = tail call <2 x double> @llvm.aarch64.neon.frint64z.v2f64(<2 x double> %a)
  ret <2 x double> %val
}
