; RUN: not llc < %s -mcpu=sm_60 -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=nvptx -mcpu=sm_60 -O0 -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc < %s -mcpu=sm_60 -o - 2> %t.err | FileCheck %s --check-prefix=FALLBACK
; RUN: FileCheck %s < %t.err

target triple = "nvptx64-unknown-cuda"

; The approximate expansion requires afn, including after promotion or
; scalarization. Other fast-math flags do not grant this permission.
; If the diagnostic handler returns, retain the existing expansion.

; CHECK-DAG: error: {{.*}}in function frem_f16 {{.*}}frem without the 'afn' fast-math flag is not supported; use the libdevice __nv_fmodf function instead
; FALLBACK-LABEL: frem_f16(
; FALLBACK: div.rn.f32
; FALLBACK: cvt.rzi.f32.f32
; FALLBACK: fma.rn.f32
define half @frem_f16(half %a, half %b) {
  %r = frem half %a, %b
  ret half %r
}

; CHECK-DAG: error: {{.*}}in function frem_f32 {{.*}}frem without the 'afn' fast-math flag is not supported; use the libdevice __nv_fmodf function instead
; FALLBACK-LABEL: frem_f32(
; FALLBACK: div.rn.f32
; FALLBACK: cvt.rzi.f32.f32
; FALLBACK: fma.rn.f32
define float @frem_f32(float %a, float %b) {
  %r = frem float %a, %b
  ret float %r
}

; CHECK-DAG: error: {{.*}}in function frem_f64 {{.*}}frem without the 'afn' fast-math flag is not supported; use the libdevice __nv_fmod function instead
; FALLBACK-LABEL: frem_f64(
; FALLBACK: div.rn.f64
; FALLBACK: cvt.rzi.f64.f64
; FALLBACK: fma.rn.f64
define double @frem_f64(double %a, double %b) {
  %r = frem double %a, %b
  ret double %r
}

; CHECK-DAG: error: {{.*}}in function frem_bf16 {{.*}}frem without the 'afn' fast-math flag is not supported; use the libdevice __nv_fmodf function instead
; FALLBACK-LABEL: frem_bf16(
; FALLBACK: div.rn.f32
; FALLBACK: cvt.rzi.f32.f32
; FALLBACK: fma.rn.f32
define bfloat @frem_bf16(bfloat %a, bfloat %b) {
  %r = frem bfloat %a, %b
  ret bfloat %r
}

; CHECK-DAG: error: {{.*}}in function frem_v2f16 {{.*}}frem without the 'afn' fast-math flag is not supported; use the libdevice __nv_fmodf function instead
define <2 x half> @frem_v2f16(<2 x half> %a, <2 x half> %b) {
  %r = frem <2 x half> %a, %b
  ret <2 x half> %r
}

; CHECK-DAG: error: {{.*}}in function frem_v2f32 {{.*}}frem without the 'afn' fast-math flag is not supported; use the libdevice __nv_fmodf function instead
define <2 x float> @frem_v2f32(<2 x float> %a, <2 x float> %b) {
  %r = frem <2 x float> %a, %b
  ret <2 x float> %r
}

; CHECK-DAG: error: {{.*}}in function frem_v2bf16 {{.*}}frem without the 'afn' fast-math flag is not supported; use the libdevice __nv_fmodf function instead
define <2 x bfloat> @frem_v2bf16(<2 x bfloat> %a, <2 x bfloat> %b) {
  %r = frem <2 x bfloat> %a, %b
  ret <2 x bfloat> %r
}

; CHECK-DAG: error: {{.*}}in function frem_other_flags {{.*}}frem without the 'afn' fast-math flag is not supported; use the libdevice __nv_fmodf function instead
define float @frem_other_flags(float %a, float %b) {
  %r = frem reassoc nnan ninf nsz arcp contract float %a, %b
  ret float %r
}
