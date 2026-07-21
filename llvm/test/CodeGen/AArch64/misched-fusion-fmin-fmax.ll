; RUN: llc %s -o - -mtriple=aarch64-linux-gnu -mattr=+fullfp16,+neon,+fuse-fmin-fmax | FileCheck %s --check-prefixes=CHECK,LINUX
; RUN: llc %s -o - -mtriple=arm64-apple-macosx -mcpu=apple-m5 | FileCheck %s --check-prefixes=CHECK,APPLE

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
; LINUX:        fmax [[R:v[0-9]+]].4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
; LINUX-NEXT:   fmax v{{[0-9]+}}.4h, [[R]].4h, v{{[0-9]+}}.4h
; APPLE:      fmax.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmax.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fmax_fmax_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.maximum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.maximum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fmax_fmin_v4f16:
; LINUX:        fmax [[R:v[0-9]+]].4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
; LINUX-NEXT:   fmin v{{[0-9]+}}.4h, [[R]].4h, v{{[0-9]+}}.4h
; APPLE:      fmax.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmin.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fmax_fmin_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.maximum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.minimum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fmin_fmax_v4f16:
; LINUX:        fmin [[R:v[0-9]+]].4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
; LINUX-NEXT:   fmax v{{[0-9]+}}.4h, [[R]].4h, v{{[0-9]+}}.4h
; APPLE:      fmin.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmax.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fmin_fmax_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.minimum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.maximum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fmin_fmin_v4f16:
; LINUX:        fmin [[R:v[0-9]+]].4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
; LINUX-NEXT:   fmin v{{[0-9]+}}.4h, [[R]].4h, v{{[0-9]+}}.4h
; APPLE:      fmin.4h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmin.4h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x half> @fmin_fmin_v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c, <4 x half> %x, <4 x half> %y) {
  %v0 = call <4 x half> @llvm.minimum.v4f16(<4 x half> %a, <4 x half> %b)
  %d = fadd <4 x half> %x, %y
  %v1 = call <4 x half> @llvm.minimum.v4f16(<4 x half> %v0, <4 x half> %c)
  %r = fadd <4 x half> %v1, %d
  ret <4 x half> %r
}

; CHECK-LABEL: fmax_fmax_v8f16:
; LINUX:        fmax [[R:v[0-9]+]].8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
; LINUX-NEXT:   fmax v{{[0-9]+}}.8h, [[R]].8h, v{{[0-9]+}}.8h
; APPLE:      fmax.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmax.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fmax_fmax_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.maximum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.maximum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fmax_fmin_v8f16:
; LINUX:        fmax [[R:v[0-9]+]].8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
; LINUX-NEXT:   fmin v{{[0-9]+}}.8h, [[R]].8h, v{{[0-9]+}}.8h
; APPLE:      fmax.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmin.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fmax_fmin_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.maximum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.minimum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fmin_fmax_v8f16:
; LINUX:        fmin [[R:v[0-9]+]].8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
; LINUX-NEXT:   fmax v{{[0-9]+}}.8h, [[R]].8h, v{{[0-9]+}}.8h
; APPLE:      fmin.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmax.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fmin_fmax_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.minimum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.maximum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fmin_fmin_v8f16:
; LINUX:        fmin [[R:v[0-9]+]].8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
; LINUX-NEXT:   fmin v{{[0-9]+}}.8h, [[R]].8h, v{{[0-9]+}}.8h
; APPLE:      fmin.8h [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmin.8h v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <8 x half> @fmin_fmin_v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, <8 x half> %x, <8 x half> %y) {
  %v0 = call <8 x half> @llvm.minimum.v8f16(<8 x half> %a, <8 x half> %b)
  %d = fadd <8 x half> %x, %y
  %v1 = call <8 x half> @llvm.minimum.v8f16(<8 x half> %v0, <8 x half> %c)
  %r = fadd <8 x half> %v1, %d
  ret <8 x half> %r
}

; CHECK-LABEL: fmax_fmax_v2f32:
; LINUX:        fmax [[R:v[0-9]+]].2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
; LINUX-NEXT:   fmax v{{[0-9]+}}.2s, [[R]].2s, v{{[0-9]+}}.2s
; APPLE:      fmax.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmax.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fmax_fmax_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.maximum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.maximum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fmax_fmin_v2f32:
; LINUX:        fmax [[R:v[0-9]+]].2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
; LINUX-NEXT:   fmin v{{[0-9]+}}.2s, [[R]].2s, v{{[0-9]+}}.2s
; APPLE:      fmax.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmin.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fmax_fmin_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.maximum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.minimum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fmin_fmax_v2f32:
; LINUX:        fmin [[R:v[0-9]+]].2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
; LINUX-NEXT:   fmax v{{[0-9]+}}.2s, [[R]].2s, v{{[0-9]+}}.2s
; APPLE:      fmin.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmax.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fmin_fmax_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.minimum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.maximum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fmin_fmin_v2f32:
; LINUX:        fmin [[R:v[0-9]+]].2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
; LINUX-NEXT:   fmin v{{[0-9]+}}.2s, [[R]].2s, v{{[0-9]+}}.2s
; APPLE:      fmin.2s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmin.2s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x float> @fmin_fmin_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, <2 x float> %x, <2 x float> %y) {
  %v0 = call <2 x float> @llvm.minimum.v2f32(<2 x float> %a, <2 x float> %b)
  %d = fadd <2 x float> %x, %y
  %v1 = call <2 x float> @llvm.minimum.v2f32(<2 x float> %v0, <2 x float> %c)
  %r = fadd <2 x float> %v1, %d
  ret <2 x float> %r
}

; CHECK-LABEL: fmax_fmax_v4f32:
; LINUX:        fmax [[R:v[0-9]+]].4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
; LINUX-NEXT:   fmax v{{[0-9]+}}.4s, [[R]].4s, v{{[0-9]+}}.4s
; APPLE:      fmax.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmax.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fmax_fmax_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.maximum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.maximum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fmax_fmin_v4f32:
; LINUX:        fmax [[R:v[0-9]+]].4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
; LINUX-NEXT:   fmin v{{[0-9]+}}.4s, [[R]].4s, v{{[0-9]+}}.4s
; APPLE:      fmax.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmin.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fmax_fmin_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.maximum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.minimum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fmin_fmax_v4f32:
; LINUX:        fmin [[R:v[0-9]+]].4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
; LINUX-NEXT:   fmax v{{[0-9]+}}.4s, [[R]].4s, v{{[0-9]+}}.4s
; APPLE:      fmin.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmax.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fmin_fmax_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.minimum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.maximum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fmin_fmin_v4f32:
; LINUX:        fmin [[R:v[0-9]+]].4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
; LINUX-NEXT:   fmin v{{[0-9]+}}.4s, [[R]].4s, v{{[0-9]+}}.4s
; APPLE:      fmin.4s [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmin.4s v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <4 x float> @fmin_fmin_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %x, <4 x float> %y) {
  %v0 = call <4 x float> @llvm.minimum.v4f32(<4 x float> %a, <4 x float> %b)
  %d = fadd <4 x float> %x, %y
  %v1 = call <4 x float> @llvm.minimum.v4f32(<4 x float> %v0, <4 x float> %c)
  %r = fadd <4 x float> %v1, %d
  ret <4 x float> %r
}

; CHECK-LABEL: fmax_fmax_v2f64:
; LINUX:        fmax [[R:v[0-9]+]].2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
; LINUX-NEXT:   fmax v{{[0-9]+}}.2d, [[R]].2d, v{{[0-9]+}}.2d
; APPLE:      fmax.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmax.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fmax_fmax_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.maximum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.maximum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}

; CHECK-LABEL: fmax_fmin_v2f64:
; LINUX:        fmax [[R:v[0-9]+]].2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
; LINUX-NEXT:   fmin v{{[0-9]+}}.2d, [[R]].2d, v{{[0-9]+}}.2d
; APPLE:      fmax.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmin.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fmax_fmin_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.maximum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.minimum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}

; CHECK-LABEL: fmin_fmax_v2f64:
; LINUX:        fmin [[R:v[0-9]+]].2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
; LINUX-NEXT:   fmax v{{[0-9]+}}.2d, [[R]].2d, v{{[0-9]+}}.2d
; APPLE:      fmin.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmax.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fmin_fmax_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.minimum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.maximum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}

; CHECK-LABEL: fmin_fmin_v2f64:
; LINUX:        fmin [[R:v[0-9]+]].2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
; LINUX-NEXT:   fmin v{{[0-9]+}}.2d, [[R]].2d, v{{[0-9]+}}.2d
; APPLE:      fmin.2d [[R:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; APPLE-NEXT: fmin.2d v{{[0-9]+}}, [[R]], v{{[0-9]+}}
define <2 x double> @fmin_fmin_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c, <2 x double> %x, <2 x double> %y) {
  %v0 = call <2 x double> @llvm.minimum.v2f64(<2 x double> %a, <2 x double> %b)
  %d = fadd <2 x double> %x, %y
  %v1 = call <2 x double> @llvm.minimum.v2f64(<2 x double> %v0, <2 x double> %c)
  %r = fadd <2 x double> %v1, %d
  ret <2 x double> %r
}
