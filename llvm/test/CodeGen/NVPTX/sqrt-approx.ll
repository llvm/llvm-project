; RUN: llc < %s -march=nvptx -mcpu=sm_20 -nvptx-prec-divf32=0 -nvptx-prec-sqrtf32=0 \
; RUN:   | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

declare float @llvm.sqrt.f32(float)
declare double @llvm.sqrt.f64(double)

; -- reciprocal sqrt --

; CHECK-LABEL: test_rsqrt32
define float @test_rsqrt32(float %a) #0 {
; CHECK: rsqrt.approx.f32
  %val = tail call float @llvm.sqrt.f32(float %a)
  %ret = fdiv float 1.0, %val
  ret float %ret
}

; CHECK-LABEL: test_rsqrt_ftz
define float @test_rsqrt_ftz(float %a) #0 #1 {
; CHECK: rsqrt.approx.ftz.f32
  %val = tail call float @llvm.sqrt.f32(float %a)
  %ret = fdiv float 1.0, %val
  ret float %ret
}

; CHECK-LABEL: test_rsqrt64
define double @test_rsqrt64(double %a) #0 {
; CHECK: rsqrt.approx.f64
  %val = tail call double @llvm.sqrt.f64(double %a)
  %ret = fdiv double 1.0, %val
  ret double %ret
}

; CHECK-LABEL: test_rsqrt64_ftz
define double @test_rsqrt64_ftz(double %a) #0 #1 {
; There's no rsqrt.approx.ftz.f64 instruction; we just use the non-ftz version.
; CHECK: rsqrt.approx.f64
  %val = tail call double @llvm.sqrt.f64(double %a)
  %ret = fdiv double 1.0, %val
  ret double %ret
}

; -- sqrt --

; CHECK-LABEL: test_sqrt32
define float @test_sqrt32(float %a) #0 {
; CHECK: sqrt.rn.f32
  %ret = tail call float @llvm.sqrt.f32(float %a)
  ret float %ret
}

; CHECK-LABEL: test_sqrt32_ninf
define float @test_sqrt32_ninf(float %a) #0 {
; CHECK: sqrt.approx.f32
  %ret = tail call ninf afn float @llvm.sqrt.f32(float %a)
  ret float %ret
}

; CHECK-LABEL: test_sqrt_ftz
define float @test_sqrt_ftz(float %a) #0 #1 {
; CHECK: sqrt.rn.ftz.f32
  %ret = tail call float @llvm.sqrt.f32(float %a)
  ret float %ret
}

; CHECK-LABEL: test_sqrt_ftz_ninf
define float @test_sqrt_ftz_ninf(float %a) #0 #1 {
; CHECK: sqrt.approx.ftz.f32
  %ret = tail call ninf afn float @llvm.sqrt.f32(float %a)
  ret float %ret
}

; CHECK-LABEL: test_sqrt64
define double @test_sqrt64(double %a) #0 {
; CHECK: sqrt.rn.f64
  %ret = tail call double @llvm.sqrt.f64(double %a)
  ret double %ret
}

; CHECK-LABEL: test_sqrt64_ninf
define double @test_sqrt64_ninf(double %a) #0 {
; There's no sqrt.approx.f64 instruction; we emit
; reciprocal(rsqrt.approx.f64(x)).  There's no non-ftz approximate reciprocal,
; so we just use the ftz version.
; CHECK: rsqrt.approx.f64
; CHECK: rcp.approx.ftz.f64
  %ret = tail call ninf afn double @llvm.sqrt.f64(double %a)
  ret double %ret
}

; CHECK-LABEL: test_sqrt64_ftz
define double @test_sqrt64_ftz(double %a) #0 #1 {
; CHECK: sqrt.rn.f64
  %ret = tail call double @llvm.sqrt.f64(double %a)
  ret double %ret
}

; CHECK-LABEL: test_sqrt64_ftz_ninf
define double @test_sqrt64_ftz_ninf(double %a) #0 #1 {
; There's no sqrt.approx.ftz.f64 instruction; we just use the non-ftz version.
; CHECK: rsqrt.approx.f64
; CHECK: rcp.approx.ftz.f64
  %ret = tail call ninf afn double @llvm.sqrt.f64(double %a)
  ret double %ret
}

; -- refined sqrt and rsqrt --
;
; The sqrt and rsqrt refinement algorithms both emit an rsqrt.approx, followed
; by some math.

; CHECK-LABEL: test_rsqrt32_refined
define float @test_rsqrt32_refined(float %a) #0 #2 {
; CHECK: rsqrt.approx.f32
  %val = tail call float @llvm.sqrt.f32(float %a)
  %ret = fdiv float 1.0, %val
  ret float %ret
}

; CHECK-LABEL: test_sqrt32_refined
define float @test_sqrt32_refined(float %a) #0 #2 {
; CHECK: sqrt.rn.f32
  %ret = tail call float @llvm.sqrt.f32(float %a)
  ret float %ret
}

; CHECK-LABEL: test_sqrt32_refined_ninf
define float @test_sqrt32_refined_ninf(float %a) #0 #2 {
; CHECK: rsqrt.approx.f32
  %ret = tail call ninf afn float @llvm.sqrt.f32(float %a)
  ret float %ret
}

; CHECK-LABEL: test_rsqrt64_refined
define double @test_rsqrt64_refined(double %a) #0 #2 {
; CHECK: rsqrt.approx.f64
  %val = tail call double @llvm.sqrt.f64(double %a)
  %ret = fdiv double 1.0, %val
  ret double %ret
}

; CHECK-LABEL: test_sqrt64_refined
define double @test_sqrt64_refined(double %a) #0 #2 {
; CHECK: sqrt.rn.f64
  %ret = tail call double @llvm.sqrt.f64(double %a)
  ret double %ret
}

; CHECK-LABEL: test_sqrt64_refined_ninf
define double @test_sqrt64_refined_ninf(double %a) #0 #2 {
; CHECK: rsqrt.approx.f64
  %ret = tail call ninf afn double @llvm.sqrt.f64(double %a)
  ret double %ret
}

; -- refined sqrt and rsqrt with ftz enabled --

; CHECK-LABEL: test_rsqrt32_refined_ftz
define float @test_rsqrt32_refined_ftz(float %a) #0 #1 #2 {
; CHECK: rsqrt.approx.ftz.f32
  %val = tail call float @llvm.sqrt.f32(float %a)
  %ret = fdiv float 1.0, %val
  ret float %ret
}

; CHECK-LABEL: test_sqrt32_refined_ftz
define float @test_sqrt32_refined_ftz(float %a) #0 #1 #2 {
; CHECK: sqrt.rn.ftz.f32
  %ret = tail call float @llvm.sqrt.f32(float %a)
  ret float %ret
}

; CHECK-LABEL: test_sqrt32_refined_ftz_ninf
define float @test_sqrt32_refined_ftz_ninf(float %a) #0 #1 #2 {
; CHECK: rsqrt.approx.ftz.f32
  %ret = tail call ninf afn float @llvm.sqrt.f32(float %a)
  ret float %ret
}

; CHECK-LABEL: test_rsqrt64_refined_ftz
define double @test_rsqrt64_refined_ftz(double %a) #0 #1 #2 {
; There's no rsqrt.approx.ftz.f64, so we just use the non-ftz version.
; CHECK: rsqrt.approx.f64
  %val = tail call double @llvm.sqrt.f64(double %a)
  %ret = fdiv double 1.0, %val
  ret double %ret
}

; CHECK-LABEL: test_sqrt64_refined_ftz
define double @test_sqrt64_refined_ftz(double %a) #0 #1 #2 {
; CHECK: sqrt.rn.f64
  %ret = tail call double @llvm.sqrt.f64(double %a)
  ret double %ret
}

; CHECK-LABEL: test_sqrt64_refined_ftz_ninf
define double @test_sqrt64_refined_ftz_ninf(double %a) #0 #1 #2 {
; CHECK: rsqrt.approx.f64
  %ret = tail call ninf afn double @llvm.sqrt.f64(double %a)
  ret double %ret
}

attributes #0 = { "unsafe-fp-math" = "true" }
attributes #1 = { "denormal-fp-math-f32" = "preserve-sign,preserve-sign" }
attributes #2 = { "reciprocal-estimates" = "rsqrtf:1,rsqrtd:1,sqrtf:1,sqrtd:1" }
