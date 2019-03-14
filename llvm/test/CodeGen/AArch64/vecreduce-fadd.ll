; RUN: llc < %s -mtriple=aarch64-eabi -aarch64-neon-syntax=generic -asm-verbose=0 -mattr=+fullfp16 | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-eabi -aarch64-neon-syntax=generic -asm-verbose=0 | FileCheck %s --check-prefix=CHECKNOFP16

define float @add_HalfS(<2 x float> %bin.rdx)  {
; CHECK-LABEL: add_HalfS:
; CHECK:       faddp s0, v0.2s
; CHECK-NEXT:  ret
  %r = call fast float @llvm.experimental.vector.reduce.fadd.f32.v2f32(<2 x float> undef, <2 x float> %bin.rdx)
  ret float %r
}

define half @add_HalfH(<4 x half> %bin.rdx)  {
; CHECK-LABEL: add_HalfH:
; CHECK:       mov   h3, v0.h[1]
; CHECK-NEXT:  mov   h1, v0.h[3]
; CHECK-NEXT:  mov   h2, v0.h[2]
; CHECK-NEXT:  fadd  h0, h0, h3
; CHECK-NEXT:  fadd  h0, h0, h2
; CHECK-NEXT:  fadd  h0, h0, h1
; CHECK-NEXT:  ret
; CHECKNOFP16-LABEL: add_HalfH:
; CHECKNOFP16-NOT:   faddp
; CHECKNOFP16-NOT:   fadd h{{[0-9]+}}
; CHECKNOFP16-NOT:   fadd v{{[0-9]+}}.{{[0-9]}}h
; CHECKNOFP16:       ret
  %r = call fast half @llvm.experimental.vector.reduce.fadd.f16.v4f16(<4 x half> undef, <4 x half> %bin.rdx)
  ret half %r
}


define half @add_H(<8 x half> %bin.rdx)  {
; CHECK-LABEL: add_H:
; CHECK:       ext   v1.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:  fadd  v0.4h, v0.4h, v1.4h
; CHECK-NEXT:  mov   h1, v0.h[1]
; CHECK-NEXT:  mov   h2, v0.h[2]
; CHECK-NEXT:  fadd  h1, h0, h1
; CHECK-NEXT:  fadd  h1, h1, h2
; CHECK-NEXT:  mov   h0, v0.h[3]
; CHECK-NEXT:  fadd  h0, h1, h0
; CHECK-NEXT:  ret

; CHECKNOFP16-LABEL: add_H:
; CHECKNOFP16-NOT:   faddp
; CHECKNOFP16-NOT:   fadd h{{[0-9]+}}
; CHECKNOFP16-NOT:   fadd v{{[0-9]+}}.{{[0-9]}}h
; CHECKNOFP16:       ret
  %r = call fast half @llvm.experimental.vector.reduce.fadd.f16.v8f16(<8 x half> undef, <8 x half> %bin.rdx)
  ret half %r
}

define float @add_S(<4 x float> %bin.rdx)  {
; CHECK-LABEL: add_S:
; CHECK:       ext   v1.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:  fadd  v0.2s, v0.2s, v1.2s
; CHECK-NEXT:  faddp s0, v0.2s
; CHECK-NEXT:  ret
  %r = call fast float @llvm.experimental.vector.reduce.fadd.f32.v4f32(<4 x float> undef, <4 x float> %bin.rdx)
  ret float %r
}

define double @add_D(<2 x double> %bin.rdx)  {
; CHECK-LABEL: add_D:
; CHECK:       faddp d0, v0.2d
; CHECK-NEXT:  ret
  %r = call fast double @llvm.experimental.vector.reduce.fadd.f64.v2f64(<2 x double> undef, <2 x double> %bin.rdx)
  ret double %r
}

define half @add_2H(<16 x half> %bin.rdx)  {
; CHECK-LABEL: add_2H:
; CHECK:       fadd  v0.8h, v0.8h, v1.8h
; CHECK-NEXT:  ext   v1.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:  fadd  v0.4h, v0.4h, v1.4h
; CHECK-NEXT:  mov   h1, v0.h[1]
; CHECK-NEXT:  mov   h2, v0.h[2]
; CHECK-NEXT:  fadd  h1, h0, h1
; CHECK-NEXT:  fadd  h1, h1, h2
; CHECK-NEXT:  mov   h0, v0.h[3]
; CHECK-NEXT:  fadd  h0, h1, h0
; CHECK-NEXT:  ret
; CHECKNOFP16-LABEL: add_2H:
; CHECKNOFP16-NOT:   faddp
; CHECKNOFP16-NOT:   fadd h{{[0-9]+}}
; CHECKNOFP16-NOT:   fadd v{{[0-9]+}}.{{[0-9]}}h
; CHECKNOFP16:       ret
  %r = call fast half @llvm.experimental.vector.reduce.fadd.f16.v16f16(<16 x half> undef, <16 x half> %bin.rdx)
  ret half %r
}

define float @add_2S(<8 x float> %bin.rdx)  {
; CHECK-LABEL: add_2S:
; CHECK:       fadd  v0.4s, v0.4s, v1.4s
; CHECK-NEXT:  ext   v1.16b, v0.16b, v0.16b, #8
; CHECK-NEXT:  fadd  v0.2s, v0.2s, v1.2s
; CHECK-NEXT:  faddp s0, v0.2s
; CHECK-NEXT:  ret
  %r = call fast float @llvm.experimental.vector.reduce.fadd.f32.v8f32(<8 x float> undef, <8 x float> %bin.rdx)
  ret float %r
}

define double @add_2D(<4 x double> %bin.rdx)  {
; CHECK-LABEL: add_2D:
; CHECK:       fadd v0.2d, v0.2d, v1.2d
; CHECK-NEXT:  faddp d0, v0.2d
; CHECK-NEXT:  ret
  %r = call fast double @llvm.experimental.vector.reduce.fadd.f64.v4f64(<4 x double> undef, <4 x double> %bin.rdx)
  ret double %r
}

; Function Attrs: nounwind readnone
declare half @llvm.experimental.vector.reduce.fadd.f16.v4f16(<4 x half>, <4 x half>)
declare half @llvm.experimental.vector.reduce.fadd.f16.v8f16(<8 x half>, <8 x half>)
declare half @llvm.experimental.vector.reduce.fadd.f16.v16f16(<16 x half>, <16 x half>)
declare float @llvm.experimental.vector.reduce.fadd.f32.v2f32(<2 x float>, <2 x float>)
declare float @llvm.experimental.vector.reduce.fadd.f32.v4f32(<4 x float>, <4 x float>)
declare float @llvm.experimental.vector.reduce.fadd.f32.v8f32(<8 x float>, <8 x float>)
declare double @llvm.experimental.vector.reduce.fadd.f64.v2f64(<2 x double>, <2 x double>)
declare double @llvm.experimental.vector.reduce.fadd.f64.v4f64(<4 x double>, <4 x double>)
