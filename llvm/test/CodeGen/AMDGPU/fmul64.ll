; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck -check-prefix=FUNC -check-prefix=SI %s
; RUN: llc -mtriple=amdgpu8.02 < %s | FileCheck -check-prefix=FUNC -check-prefix=SI %s

; FUNC-LABEL: {{^}}fmul_f64:
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define double @fmul_f64(double %in1, double %in2) {
   %r2 = fmul double %in1, %in2
  ret double %r2
}

; FUNC-LABEL: {{^}}fmul_v2f64:
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define <2 x double> @fmul_v2f64(<2 x double> %in1, <2 x double> %in2) {
   %r2 = fmul <2 x double> %in1, %in2
  ret <2 x double> %r2
}

; FUNC-LABEL: {{^}}fmul_v4f64:
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\]}}
define <4 x double> @fmul_v4f64(<4 x double> %in1, <4 x double> %in2) {
   %r2 = fmul <4 x double> %in1, %in2
  ret <4 x double> %r2
}
