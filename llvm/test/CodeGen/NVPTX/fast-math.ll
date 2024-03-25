; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

declare float @llvm.sqrt.f32(float)
declare double @llvm.sqrt.f64(double)

; CHECK-LABEL: sqrt_div(
; CHECK: sqrt.rn.f32
; CHECK: div.rn.f32
define float @sqrt_div(float %a, float %b) {
  %t1 = tail call float @llvm.sqrt.f32(float %a)
  %t2 = fdiv float %t1, %b
  ret float %t2
}

; CHECK-LABEL: sqrt_div_fast(
; CHECK: sqrt.rn.f32
; CHECK: div.approx.f32
define float @sqrt_div_fast(float %a, float %b) #0 {
  %t1 = tail call float @llvm.sqrt.f32(float %a)
  %t2 = fdiv float %t1, %b
  ret float %t2
}

; CHECK-LABEL: sqrt_div_fast_ninf(
; CHECK: sqrt.approx.f32
; CHECK: div.approx.f32
define float @sqrt_div_fast_ninf(float %a, float %b) #0 {
  %t1 = tail call ninf afn float @llvm.sqrt.f32(float %a)
  %t2 = fdiv float %t1, %b
  ret float %t2
}

; CHECK-LABEL: sqrt_div_ftz(
; CHECK: sqrt.rn.ftz.f32
; CHECK: div.rn.ftz.f32
define float @sqrt_div_ftz(float %a, float %b) #1 {
  %t1 = tail call float @llvm.sqrt.f32(float %a)
  %t2 = fdiv float %t1, %b
  ret float %t2
}

; CHECK-LABEL: sqrt_div_fast_ftz(
; CHECK: sqrt.rn.ftz.f32
; CHECK: div.approx.ftz.f32
define float @sqrt_div_fast_ftz(float %a, float %b) #0 #1 {
  %t1 = tail call float @llvm.sqrt.f32(float %a)
  %t2 = fdiv float %t1, %b
  ret float %t2
}

; CHECK-LABEL: sqrt_div_fast_ftz_ninf(
; CHECK: sqrt.approx.ftz.f32
; CHECK: div.approx.ftz.f32
define float @sqrt_div_fast_ftz_ninf(float %a, float %b) #0 #1 {
  %t1 = tail call ninf afn float @llvm.sqrt.f32(float %a)
  %t2 = fdiv float %t1, %b
  ret float %t2
}

; There are no fast-math or ftz versions of sqrt and div for f64.  We use
; reciprocal(rsqrt(x)) for sqrt(x), and emit a vanilla divide.

; CHECK-LABEL: sqrt_div_fast_ftz_f64(
; CHECK: sqrt.rn.f64
; CHECK: div.rn.f64
define double @sqrt_div_fast_ftz_f64(double %a, double %b) #0 #1 {
  %t1 = tail call double @llvm.sqrt.f64(double %a)
  %t2 = fdiv double %t1, %b
  ret double %t2
}

; CHECK-LABEL: sqrt_div_fast_ftz_f64_ninf(
; CHECK: rsqrt.approx.f64
; CHECK: rcp.approx.ftz.f64
; CHECK: div.rn.f64
define double @sqrt_div_fast_ftz_f64_ninf(double %a, double %b) #0 #1 {
  %t1 = tail call ninf afn double @llvm.sqrt.f64(double %a)
  %t2 = fdiv double %t1, %b
  ret double %t2
}

; CHECK-LABEL: rsqrt(
; CHECK-NOT: rsqrt.approx
; CHECK: sqrt.rn.f32
; CHECK-NOT: rsqrt.approx
define float @rsqrt(float %a) {
  %b = tail call float @llvm.sqrt.f32(float %a)
  %ret = fdiv float 1.0, %b
  ret float %ret
}

; CHECK-LABEL: rsqrt_fast(
; CHECK-NOT: div.
; CHECK-NOT: sqrt.
; CHECK: rsqrt.approx.f32
; CHECK-NOT: div.
; CHECK-NOT: sqrt.
define float @rsqrt_fast(float %a) #0 {
  %b = tail call float @llvm.sqrt.f32(float %a)
  %ret = fdiv float 1.0, %b
  ret float %ret
}

; CHECK-LABEL: rsqrt_fast_ftz(
; CHECK-NOT: div.
; CHECK-NOT: sqrt.
; CHECK: rsqrt.approx.ftz.f32
; CHECK-NOT: div.
; CHECK-NOT: sqrt.
define float @rsqrt_fast_ftz(float %a) #0 #1 {
  %b = tail call float @llvm.sqrt.f32(float %a)
  %ret = fdiv float 1.0, %b
  ret float %ret
}

; CHECK-LABEL: fadd
; CHECK: add.rn.f32
define float @fadd(float %a, float %b) {
  %t1 = fadd float %a, %b
  ret float %t1
}

; CHECK-LABEL: fadd_ftz
; CHECK: add.rn.ftz.f32
define float @fadd_ftz(float %a, float %b) #1 {
  %t1 = fadd float %a, %b
  ret float %t1
}

declare float @llvm.sin.f32(float)
declare float @llvm.cos.f32(float)

; CHECK-LABEL: fsin_approx
; CHECK:       sin.approx.f32
define float @fsin_approx(float %a) #0 {
  %r = tail call float @llvm.sin.f32(float %a)
  ret float %r
}

; CHECK-LABEL: fcos_approx
; CHECK:       cos.approx.f32
define float @fcos_approx(float %a) #0 {
  %r = tail call float @llvm.cos.f32(float %a)
  ret float %r
}

; CHECK-LABEL: repeated_div_recip_allowed
define float @repeated_div_recip_allowed(i1 %pred, float %a, float %b, float %divisor) {
; CHECK: rcp.rn.f32
; CHECK: mul.rn.f32
; CHECK: mul.rn.f32
; CHECK: mul.rn.f32
; CHECK: selp.f32
  %x = fdiv arcp float %a, %divisor
  %y = fdiv arcp float %b, %divisor
  %z = fmul float %x, %y
  %w = select i1 %pred, float %z, float %y
  ret float %w
}

; CHECK-LABEL: repeated_div_recip_allowed_sel
define float @repeated_div_recip_allowed_sel(i1 %pred, float %a, float %b, float %divisor) {
; CHECK: selp.f32
; CHECK: div.rn.f32
  %x = fdiv arcp float %a, %divisor
  %y = fdiv arcp float %b, %divisor
  %w = select i1 %pred, float %x, float %y
  ret float %w
}

; CHECK-LABEL: repeated_div_recip_allowed_ftz
define float @repeated_div_recip_allowed_ftz(i1 %pred, float %a, float %b, float %divisor) #1 {
; CHECK: rcp.rn.ftz.f32
; CHECK: mul.rn.ftz.f32
; CHECK: mul.rn.ftz.f32
; CHECK: mul.rn.ftz.f32
; CHECK: selp.f32
  %x = fdiv arcp float %a, %divisor
  %y = fdiv arcp float %b, %divisor
  %z = fmul float %x, %y
  %w = select i1 %pred, float %z, float %y
  ret float %w
}

; CHECK-LABEL: repeated_div_recip_allowed_ftz_sel
define float @repeated_div_recip_allowed_ftz_sel(i1 %pred, float %a, float %b, float %divisor) #1 {
; CHECK: selp.f32
; CHECK: div.rn.ftz.f32
  %x = fdiv arcp float %a, %divisor
  %y = fdiv arcp float %b, %divisor
  %w = select i1 %pred, float %x, float %y
  ret float %w
}

; CHECK-LABEL: repeated_div_fast
define float @repeated_div_fast(i1 %pred, float %a, float %b, float %divisor) #0 {
; CHECK: rcp.approx.f32
; CHECK: mul.f32
; CHECK: mul.f32
; CHECK: mul.f32
; CHECK: selp.f32
  %x = fdiv float %a, %divisor
  %y = fdiv float %b, %divisor
  %z = fmul float %x, %y
  %w = select i1 %pred, float %z, float %y
  ret float %w
}

; CHECK-LABEL: repeated_div_fast_sel
define float @repeated_div_fast_sel(i1 %pred, float %a, float %b, float %divisor) #0 {
; CHECK: selp.f32
; CHECK: div.approx.f32
  %x = fdiv float %a, %divisor
  %y = fdiv float %b, %divisor
  %w = select i1 %pred, float %x, float %y
  ret float %w
}

; CHECK-LABEL: repeated_div_fast_ftz
define float @repeated_div_fast_ftz(i1 %pred, float %a, float %b, float %divisor) #0 #1 {
; CHECK: rcp.approx.ftz.f32
; CHECK: mul.ftz.f32
; CHECK: mul.ftz.f32
; CHECK: mul.ftz.f32
; CHECK: selp.f32
  %x = fdiv float %a, %divisor
  %y = fdiv float %b, %divisor
  %z = fmul float %x, %y
  %w = select i1 %pred, float %z, float %y
  ret float %w
}

; CHECK-LABEL: repeated_div_fast_ftz_sel
define float @repeated_div_fast_ftz_sel(i1 %pred, float %a, float %b, float %divisor) #0 #1 {
; CHECK: selp.f32
; CHECK: div.approx.ftz.f32
  %x = fdiv float %a, %divisor
  %y = fdiv float %b, %divisor
  %w = select i1 %pred, float %x, float %y
  ret float %w
}

; CHECK-LABEL: frem
define float @frem(float %a, float %b) #0 {
  ; CHECK-NOT: testp.infinite
  %rem = frem float %a, %b
  ret float %rem
}

; CHECK-LABEL: frem_ftz
define float @frem_ftz(float %a, float %b) #0 #1 {
  ; CHECK-NOT: testp.infinite
  %rem = frem float %a, %b
  ret float %rem
}

; CHECK-LABEL: frem_f64
define double @frem_f64(double %a, double %b) #0 {
  ; CHECK-NOT: testp.infinite
  %rem = frem double %a, %b
  ret double %rem
}

attributes #0 = { "unsafe-fp-math" = "true" }
attributes #1 = { "denormal-fp-math-f32" = "preserve-sign" }
