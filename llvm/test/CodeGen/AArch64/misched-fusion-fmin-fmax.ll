; RUN: llc %s -o - -mtriple=arm64-apple-macosx -mcpu=apple-m4 -mattr=+fuse-fmin-fmax | FileCheck %s --check-prefixes=CHECK
; RUN: llc %s -o - -mtriple=arm64-apple-macosx -mcpu=apple-m5 | FileCheck %s --check-prefixes=CHECK

declare half @llvm.maximum.f16(half, half)
declare half @llvm.minimum.f16(half, half)
declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
declare double @llvm.maximum.f64(double, double)
declare double @llvm.minimum.f64(double, double)
declare <4 x half> @llvm.maximum.v4f16(<4 x half>, <4 x half>)
declare <4 x half> @llvm.minimum.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.maximum.v8f16(<8 x half>, <8 x half>)
declare <8 x half> @llvm.minimum.v8f16(<8 x half>, <8 x half>)
declare <2 x float> @llvm.maximum.v2f32(<2 x float>, <2 x float>)
declare <2 x float> @llvm.minimum.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.maximum.v4f32(<4 x float>, <4 x float>)
declare <4 x float> @llvm.minimum.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.maximum.v2f64(<2 x double>, <2 x double>)
declare <2 x double> @llvm.minimum.v2f64(<2 x double>, <2 x double>)
declare half @llvm.maxnum.f16(half, half)
declare half @llvm.minnum.f16(half, half)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)
declare double @llvm.maxnum.f64(double, double)
declare double @llvm.minnum.f64(double, double)
declare <4 x half> @llvm.maxnum.v4f16(<4 x half>, <4 x half>)
declare <4 x half> @llvm.minnum.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.maxnum.v8f16(<8 x half>, <8 x half>)
declare <8 x half> @llvm.minnum.v8f16(<8 x half>, <8 x half>)
declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>)
declare <2 x float> @llvm.minnum.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.maxnum.v2f64(<2 x double>, <2 x double>)
declare <2 x double> @llvm.minnum.v2f64(<2 x double>, <2 x double>)

; CHECK-LABEL: fmax_fmax_h:
; CHECK:      fmax [[R:h[0-9]+]], h{{[0-9]+}}, h{{[0-9]+}}
; CHECK-NEXT: fmax h{{[0-9]+}}, [[R]], h{{[0-9]+}}
define half @fmax_fmax_h(half %a, half %b, half %c, half %x, half %y) {
  %v0 = call half @llvm.maximum.f16(half %a, half %b)
  %d = fadd half %x, %y
  %v1 = call half @llvm.maximum.f16(half %v0, half %c)
  %r = fadd half %v1, %d
  ret half %r
}

; CHECK-LABEL: fmax_fmin_h:
; CHECK:      fmax [[R:h[0-9]+]], h{{[0-9]+}}, h{{[0-9]+}}
; CHECK-NEXT: fmin h{{[0-9]+}}, [[R]], h{{[0-9]+}}
define half @fmax_fmin_h(half %a, half %b, half %c, half %x, half %y) {
  %v0 = call half @llvm.maximum.f16(half %a, half %b)
  %d = fadd half %x, %y
  %v1 = call half @llvm.minimum.f16(half %v0, half %c)
  %r = fadd half %v1, %d
  ret half %r
}

; CHECK-LABEL: fmin_fmax_h:
; CHECK:      fmin [[R:h[0-9]+]], h{{[0-9]+}}, h{{[0-9]+}}
; CHECK-NEXT: fmax h{{[0-9]+}}, [[R]], h{{[0-9]+}}
define half @fmin_fmax_h(half %a, half %b, half %c, half %x, half %y) {
  %v0 = call half @llvm.minimum.f16(half %a, half %b)
  %d = fadd half %x, %y
  %v1 = call half @llvm.maximum.f16(half %v0, half %c)
  %r = fadd half %v1, %d
  ret half %r
}

; CHECK-LABEL: fmin_fmin_h:
; CHECK:      fmin [[R:h[0-9]+]], h{{[0-9]+}}, h{{[0-9]+}}
; CHECK-NEXT: fmin h{{[0-9]+}}, [[R]], h{{[0-9]+}}
define half @fmin_fmin_h(half %a, half %b, half %c, half %x, half %y) {
  %v0 = call half @llvm.minimum.f16(half %a, half %b)
  %d = fadd half %x, %y
  %v1 = call half @llvm.minimum.f16(half %v0, half %c)
  %r = fadd half %v1, %d
  ret half %r
}

; CHECK-LABEL: fmax_fmax_s:
; CHECK:      fmax [[R:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NEXT: fmax s{{[0-9]+}}, [[R]], s{{[0-9]+}}
define float @fmax_fmax_s(float %a, float %b, float %c, float %x, float %y) {
  %v0 = call float @llvm.maximum.f32(float %a, float %b)
  %d = fadd float %x, %y
  %v1 = call float @llvm.maximum.f32(float %v0, float %c)
  %r = fadd float %v1, %d
  ret float %r
}

; CHECK-LABEL: fmax_fmin_s:
; CHECK:      fmax [[R:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NEXT: fmin s{{[0-9]+}}, [[R]], s{{[0-9]+}}
define float @fmax_fmin_s(float %a, float %b, float %c, float %x, float %y) {
  %v0 = call float @llvm.maximum.f32(float %a, float %b)
  %d = fadd float %x, %y
  %v1 = call float @llvm.minimum.f32(float %v0, float %c)
  %r = fadd float %v1, %d
  ret float %r
}

; CHECK-LABEL: fmin_fmax_s:
; CHECK:      fmin [[R:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NEXT: fmax s{{[0-9]+}}, [[R]], s{{[0-9]+}}
define float @fmin_fmax_s(float %a, float %b, float %c, float %x, float %y) {
  %v0 = call float @llvm.minimum.f32(float %a, float %b)
  %d = fadd float %x, %y
  %v1 = call float @llvm.maximum.f32(float %v0, float %c)
  %r = fadd float %v1, %d
  ret float %r
}

; CHECK-LABEL: fmin_fmin_s:
; CHECK:      fmin [[R:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NEXT: fmin s{{[0-9]+}}, [[R]], s{{[0-9]+}}
define float @fmin_fmin_s(float %a, float %b, float %c, float %x, float %y) {
  %v0 = call float @llvm.minimum.f32(float %a, float %b)
  %d = fadd float %x, %y
  %v1 = call float @llvm.minimum.f32(float %v0, float %c)
  %r = fadd float %v1, %d
  ret float %r
}

; CHECK-LABEL: fmax_fmax_d:
; CHECK:      fmax [[R:d[0-9]+]], d{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NEXT: fmax d{{[0-9]+}}, [[R]], d{{[0-9]+}}
define double @fmax_fmax_d(double %a, double %b, double %c, double %x, double %y) {
  %v0 = call double @llvm.maximum.f64(double %a, double %b)
  %d = fadd double %x, %y
  %v1 = call double @llvm.maximum.f64(double %v0, double %c)
  %r = fadd double %v1, %d
  ret double %r
}

; CHECK-LABEL: fmax_fmin_d:
; CHECK:      fmax [[R:d[0-9]+]], d{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NEXT: fmin d{{[0-9]+}}, [[R]], d{{[0-9]+}}
define double @fmax_fmin_d(double %a, double %b, double %c, double %x, double %y) {
  %v0 = call double @llvm.maximum.f64(double %a, double %b)
  %d = fadd double %x, %y
  %v1 = call double @llvm.minimum.f64(double %v0, double %c)
  %r = fadd double %v1, %d
  ret double %r
}

; CHECK-LABEL: fmin_fmax_d:
; CHECK:      fmin [[R:d[0-9]+]], d{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NEXT: fmax d{{[0-9]+}}, [[R]], d{{[0-9]+}}
define double @fmin_fmax_d(double %a, double %b, double %c, double %x, double %y) {
  %v0 = call double @llvm.minimum.f64(double %a, double %b)
  %d = fadd double %x, %y
  %v1 = call double @llvm.maximum.f64(double %v0, double %c)
  %r = fadd double %v1, %d
  ret double %r
}

; CHECK-LABEL: fmin_fmin_d:
; CHECK:      fmin [[R:d[0-9]+]], d{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NEXT: fmin d{{[0-9]+}}, [[R]], d{{[0-9]+}}
define double @fmin_fmin_d(double %a, double %b, double %c, double %x, double %y) {
  %v0 = call double @llvm.minimum.f64(double %a, double %b)
  %d = fadd double %x, %y
  %v1 = call double @llvm.minimum.f64(double %v0, double %c)
  %r = fadd double %v1, %d
  ret double %r
}

; CHECK-LABEL: fmax_fmax_v4f16:
; CHECK:      fmax.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmax.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fmax_fmax_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.maximum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.maximum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fmax_fmin_v4f16:
; CHECK:      fmax.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmin.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fmax_fmin_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.maximum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.minimum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fmin_fmax_v4f16:
; CHECK:      fmin.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmax.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fmin_fmax_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.minimum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.maximum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fmin_fmin_v4f16:
; CHECK:      fmin.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmin.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fmin_fmin_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.minimum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.minimum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fmax_fmax_v8f16:
; CHECK:      fmax.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmax.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fmax_fmax_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.maximum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.maximum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fmax_fmin_v8f16:
; CHECK:      fmax.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmin.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fmax_fmin_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.maximum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.minimum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fmin_fmax_v8f16:
; CHECK:      fmin.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmax.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fmin_fmax_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.minimum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.maximum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fmin_fmin_v8f16:
; CHECK:      fmin.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmin.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fmin_fmin_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.minimum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.minimum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fmax_fmax_v2f32:
; CHECK:      fmax.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmax.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fmax_fmax_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.maximum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.maximum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fmax_fmin_v2f32:
; CHECK:      fmax.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmin.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fmax_fmin_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.maximum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.minimum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fmin_fmax_v2f32:
; CHECK:      fmin.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmax.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fmin_fmax_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.minimum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.maximum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fmin_fmin_v2f32:
; CHECK:      fmin.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmin.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fmin_fmin_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.minimum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.minimum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fmax_fmax_v4f32:
; CHECK:      fmax.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmax.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fmax_fmax_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.maximum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.maximum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fmax_fmin_v4f32:
; CHECK:      fmax.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmin.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fmax_fmin_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.maximum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.minimum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fmin_fmax_v4f32:
; CHECK:      fmin.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmax.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fmin_fmax_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.minimum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.maximum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fmin_fmin_v4f32:
; CHECK:      fmin.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmin.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fmin_fmin_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.minimum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.minimum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fmax_fmax_v2f64:
; CHECK:      fmax.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmax.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fmax_fmax_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.maximum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.maximum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}

; CHECK-LABEL: fmax_fmin_v2f64:
; CHECK:      fmax.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmin.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fmax_fmin_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.maximum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.minimum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}

; CHECK-LABEL: fmin_fmax_v2f64:
; CHECK:      fmin.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmax.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fmin_fmax_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.minimum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.maximum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}

; CHECK-LABEL: fmin_fmin_v2f64:
; CHECK:      fmin.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmin.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fmin_fmin_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.minimum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.minimum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}

; CHECK-LABEL: fmaxnm_fmaxnm_h:
; CHECK:      fmaxnm [[R:h[0-9]+]], h{{[0-9]+}}, h{{[0-9]+}}
; CHECK-NEXT: fmaxnm h{{[0-9]+}}, [[R]], h{{[0-9]+}}
define half @fmaxnm_fmaxnm_h(half %a, half %b, half %c, half %x, half %y) {
  %v0 = call half @llvm.maxnum.f16(half %a, half %b)
  %d = fadd half %x, %y
  %v1 = call half @llvm.maxnum.f16(half %v0, half %c)
  %r = fadd half %v1, %d
  ret half %r
}

; CHECK-LABEL: fmaxnm_fminnm_h:
; CHECK:      fmaxnm [[R:h[0-9]+]], h{{[0-9]+}}, h{{[0-9]+}}
; CHECK-NEXT: fminnm h{{[0-9]+}}, [[R]], h{{[0-9]+}}
define half @fmaxnm_fminnm_h(half %a, half %b, half %c, half %x, half %y) {
  %v0 = call half @llvm.maxnum.f16(half %a, half %b)
  %d = fadd half %x, %y
  %v1 = call half @llvm.minnum.f16(half %v0, half %c)
  %r = fadd half %v1, %d
  ret half %r
}

; CHECK-LABEL: fminnm_fmaxnm_h:
; CHECK:      fminnm [[R:h[0-9]+]], h{{[0-9]+}}, h{{[0-9]+}}
; CHECK-NEXT: fmaxnm h{{[0-9]+}}, [[R]], h{{[0-9]+}}
define half @fminnm_fmaxnm_h(half %a, half %b, half %c, half %x, half %y) {
  %v0 = call half @llvm.minnum.f16(half %a, half %b)
  %d = fadd half %x, %y
  %v1 = call half @llvm.maxnum.f16(half %v0, half %c)
  %r = fadd half %v1, %d
  ret half %r
}

; CHECK-LABEL: fminnm_fminnm_h:
; CHECK:      fminnm [[R:h[0-9]+]], h{{[0-9]+}}, h{{[0-9]+}}
; CHECK-NEXT: fminnm h{{[0-9]+}}, [[R]], h{{[0-9]+}}
define half @fminnm_fminnm_h(half %a, half %b, half %c, half %x, half %y) {
  %v0 = call half @llvm.minnum.f16(half %a, half %b)
  %d = fadd half %x, %y
  %v1 = call half @llvm.minnum.f16(half %v0, half %c)
  %r = fadd half %v1, %d
  ret half %r
}

; CHECK-LABEL: fmaxnm_fmaxnm_s:
; CHECK:      fmaxnm [[R:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NEXT: fmaxnm s{{[0-9]+}}, [[R]], s{{[0-9]+}}
define float @fmaxnm_fmaxnm_s(float %a, float %b, float %c, float %x, float %y) {
  %v0 = call float @llvm.maxnum.f32(float %a, float %b)
  %d = fadd float %x, %y
  %v1 = call float @llvm.maxnum.f32(float %v0, float %c)
  %r = fadd float %v1, %d
  ret float %r
}

; CHECK-LABEL: fmaxnm_fminnm_s:
; CHECK:      fmaxnm [[R:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NEXT: fminnm s{{[0-9]+}}, [[R]], s{{[0-9]+}}
define float @fmaxnm_fminnm_s(float %a, float %b, float %c, float %x, float %y) {
  %v0 = call float @llvm.maxnum.f32(float %a, float %b)
  %d = fadd float %x, %y
  %v1 = call float @llvm.minnum.f32(float %v0, float %c)
  %r = fadd float %v1, %d
  ret float %r
}

; CHECK-LABEL: fminnm_fmaxnm_s:
; CHECK:      fminnm [[R:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NEXT: fmaxnm s{{[0-9]+}}, [[R]], s{{[0-9]+}}
define float @fminnm_fmaxnm_s(float %a, float %b, float %c, float %x, float %y) {
  %v0 = call float @llvm.minnum.f32(float %a, float %b)
  %d = fadd float %x, %y
  %v1 = call float @llvm.maxnum.f32(float %v0, float %c)
  %r = fadd float %v1, %d
  ret float %r
}

; CHECK-LABEL: fminnm_fminnm_s:
; CHECK:      fminnm [[R:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NEXT: fminnm s{{[0-9]+}}, [[R]], s{{[0-9]+}}
define float @fminnm_fminnm_s(float %a, float %b, float %c, float %x, float %y) {
  %v0 = call float @llvm.minnum.f32(float %a, float %b)
  %d = fadd float %x, %y
  %v1 = call float @llvm.minnum.f32(float %v0, float %c)
  %r = fadd float %v1, %d
  ret float %r
}

; CHECK-LABEL: fmaxnm_fmaxnm_d:
; CHECK:      fmaxnm [[R:d[0-9]+]], d{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NEXT: fmaxnm d{{[0-9]+}}, [[R]], d{{[0-9]+}}
define double @fmaxnm_fmaxnm_d(double %a, double %b, double %c, double %x, double %y) {
  %v0 = call double @llvm.maxnum.f64(double %a, double %b)
  %d = fadd double %x, %y
  %v1 = call double @llvm.maxnum.f64(double %v0, double %c)
  %r = fadd double %v1, %d
  ret double %r
}

; CHECK-LABEL: fmaxnm_fminnm_d:
; CHECK:      fmaxnm [[R:d[0-9]+]], d{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NEXT: fminnm d{{[0-9]+}}, [[R]], d{{[0-9]+}}
define double @fmaxnm_fminnm_d(double %a, double %b, double %c, double %x, double %y) {
  %v0 = call double @llvm.maxnum.f64(double %a, double %b)
  %d = fadd double %x, %y
  %v1 = call double @llvm.minnum.f64(double %v0, double %c)
  %r = fadd double %v1, %d
  ret double %r
}

; CHECK-LABEL: fminnm_fmaxnm_d:
; CHECK:      fminnm [[R:d[0-9]+]], d{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NEXT: fmaxnm d{{[0-9]+}}, [[R]], d{{[0-9]+}}
define double @fminnm_fmaxnm_d(double %a, double %b, double %c, double %x, double %y) {
  %v0 = call double @llvm.minnum.f64(double %a, double %b)
  %d = fadd double %x, %y
  %v1 = call double @llvm.maxnum.f64(double %v0, double %c)
  %r = fadd double %v1, %d
  ret double %r
}

; CHECK-LABEL: fminnm_fminnm_d:
; CHECK:      fminnm [[R:d[0-9]+]], d{{[0-9]+}}, d{{[0-9]+}}
; CHECK-NEXT: fminnm d{{[0-9]+}}, [[R]], d{{[0-9]+}}
define double @fminnm_fminnm_d(double %a, double %b, double %c, double %x, double %y) {
  %v0 = call double @llvm.minnum.f64(double %a, double %b)
  %d = fadd double %x, %y
  %v1 = call double @llvm.minnum.f64(double %v0, double %c)
  %r = fadd double %v1, %d
  ret double %r
}

; CHECK-LABEL: fmaxnm_fmaxnm_v4f16:
; CHECK:      fmaxnm.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmaxnm.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fmaxnm_fmaxnm_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.maxnum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.maxnum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fmaxnm_fminnm_v4f16:
; CHECK:      fmaxnm.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fminnm.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fmaxnm_fminnm_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.maxnum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.minnum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fminnm_fmaxnm_v4f16:
; CHECK:      fminnm.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmaxnm.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fminnm_fmaxnm_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.minnum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.maxnum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fminnm_fminnm_v4f16:
; CHECK:      fminnm.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fminnm.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fminnm_fminnm_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.minnum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.minnum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fmaxnm_fmaxnm_v8f16:
; CHECK:      fmaxnm.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmaxnm.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fmaxnm_fmaxnm_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.maxnum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.maxnum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fmaxnm_fminnm_v8f16:
; CHECK:      fmaxnm.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fminnm.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fmaxnm_fminnm_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.maxnum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.minnum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fminnm_fmaxnm_v8f16:
; CHECK:      fminnm.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmaxnm.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fminnm_fmaxnm_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.minnum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.maxnum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fminnm_fminnm_v8f16:
; CHECK:      fminnm.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fminnm.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fminnm_fminnm_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.minnum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.minnum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fmaxnm_fmaxnm_v2f32:
; CHECK:      fmaxnm.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmaxnm.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fmaxnm_fmaxnm_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fmaxnm_fminnm_v2f32:
; CHECK:      fmaxnm.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fminnm.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fmaxnm_fminnm_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.minnum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fminnm_fmaxnm_v2f32:
; CHECK:      fminnm.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmaxnm.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fminnm_fmaxnm_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.minnum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fminnm_fminnm_v2f32:
; CHECK:      fminnm.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fminnm.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fminnm_fminnm_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.minnum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.minnum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fmaxnm_fmaxnm_v4f32:
; CHECK:      fmaxnm.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmaxnm.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fmaxnm_fmaxnm_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fmaxnm_fminnm_v4f32:
; CHECK:      fmaxnm.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fminnm.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fmaxnm_fminnm_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.minnum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fminnm_fmaxnm_v4f32:
; CHECK:      fminnm.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmaxnm.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fminnm_fmaxnm_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.minnum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fminnm_fminnm_v4f32:
; CHECK:      fminnm.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fminnm.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fminnm_fminnm_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.minnum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.minnum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fmaxnm_fmaxnm_v2f64:
; CHECK:      fmaxnm.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmaxnm.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fmaxnm_fmaxnm_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}

; CHECK-LABEL: fmaxnm_fminnm_v2f64:
; CHECK:      fmaxnm.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fminnm.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fmaxnm_fminnm_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.minnum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}

; CHECK-LABEL: fminnm_fmaxnm_v2f64:
; CHECK:      fminnm.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmaxnm.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fminnm_fmaxnm_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.minnum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}

; CHECK-LABEL: fminnm_fminnm_v2f64:
; CHECK:      fminnm.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fminnm.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fminnm_fminnm_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.minnum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.minnum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}

;; Scalar and vector instructions should not be mixed
; CHECK-LABEL: fmax_fmin_mixed_scalar_vector:
; CHECK:      fmax.4s v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fadd.4s v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NEXT: fmin s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define float @fmax_fmin_mixed_scalar_vector(<4 x float> %a, <4 x float> %b, float %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.maximum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %e = extractelement <4 x float> %v0, i32 0
  %v1 = call float @llvm.minimum.f32(float %e, float %c)
  %f = extractelement <4 x float> %d, i32 0
  %r = fadd float %v1, %f
  ret float %r
}

;; FMIN/FMAX and FMINNM/FMAXNM should not be mixed
; CHECK-LABEL: fmax_fminnm_mixed:
; CHECK:      fmax s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NEXT: fadd s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; CHECK-NEXT: fminnm s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define float @fmax_fminnm_mixed(float %a, float %b, float %c, float %x, float %y) {
  %v0 = call float @llvm.maximum.f32(float %a, float %b)
  %d = fadd float %x, %y
  %v1 = call float @llvm.minnum.f32(float %v0, float %c)
  %r = fadd float %v1, %d
  ret float %r
}
