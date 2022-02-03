; RUN: llc -O3 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -O3 -mtriple=powerpc-ibm-aix-xcoff < %s | FileCheck %s

declare float @llvm.cos.f32(float)
declare float @llvm.exp.f32(float)
declare float @llvm.log10.f32(float)
declare float @llvm.log.f32(float)
declare float @llvm.pow.f32(float, float)
declare float @llvm.rint.f32(float)
declare float @llvm.sin.f32(float)
declare double @llvm.cos.f64(double)
declare double @llvm.exp.f64(double)
declare double @llvm.log.f64(double)
declare double @llvm.log10.f64(double)
declare double @llvm.pow.f64(double, double)
declare double @llvm.sin.f64(double)

; With fast-math flag specified per-function
define float @cosf_f32(float %a) {
; CHECK-LABEL: cosf_f32
; CHECK: __xl_cosf_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz float @llvm.cos.f32(float %a)
  ret float %0
}

; With fast-math flag specified per-function
define float @expf_f32(float %a) {
; CHECK-LABEL: expf_f32
; CHECK: __xl_expf_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz float @llvm.exp.f32(float %a)
  ret float %0
}

; With fast-math flag specified per-function
define float @log10f_f32(float %a) {
; CHECK-LABEL: log10f_f32
; CHECK: __xl_log10f_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz float @llvm.log10.f32(float %a)
  ret float %0
}

; With fast-math flag specified per-function
define float @logf_f32(float %a) {
; CHECK-LABEL: logf_f32
; CHECK: __xl_logf_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz float @llvm.log.f32(float %a)
  ret float %0
}

; With fast-math flag specified per-function
define float @powf_f32(float %a, float %b) {
; CHECK-LABEL: powf_f32
; CHECK: __xl_powf_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz float @llvm.pow.f32(float %a, float %b)
  ret float %0
}

; With fast-math flag specified per-function
define float @rintf_f32(float %a) {
; CHECK-LABEL: rintf_f32
; CHECK-NOT: bl __xl_rintf_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz float @llvm.rint.f32(float %a)
  ret float %0
}

; With fast-math flag specified per-function
define float @sinf_f32(float %a) {
; CHECK-LABEL: sinf_f32
; CHECK: __xl_sinf_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz float @llvm.sin.f32(float %a)
  ret float %0
}

; With fast-math flag specified per-function
define double @cos_f64(double %a) {
; CHECK-LABEL: cos_f64
; CHECK: __xl_cos_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz double @llvm.cos.f64(double %a)
  ret double %0
}

; With fast-math flag specified per-function
define double @exp_f64(double %a) {
; CHECK-LABEL: exp_f64
; CHECK: __xl_exp_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz double @llvm.exp.f64(double %a)
  ret double %0
}

; With fast-math flag specified per-function
define double @log_f64(double %a) {
; CHECK-LABEL: log_f64
; CHECK: __xl_log_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz double @llvm.log.f64(double %a)
  ret double %0
}

; With fast-math flag specified per-function
define double @log10_f64(double %a) {
; CHECK-LABEL: log10_f64
; CHECK: __xl_log10_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz double @llvm.log10.f64(double %a)
  ret double %0
}

; With fast-math flag specified per-function
define double @pow_f64(double %a, double %b) {
; CHECK-LABEL: pow_f64
; CHECK: __xl_pow_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz double @llvm.pow.f64(double %a, double %b)
  ret double %0
}

; With fast-math flag specified per-function
define double @sin_f64(double %a) {
; CHECK-LABEL: sin_f64
; CHECK: __xl_sin_finite
; CHECK: blr
entry:
  %0 = tail call nnan ninf afn nsz double @llvm.sin.f64(double %a)
  ret double %0
}
