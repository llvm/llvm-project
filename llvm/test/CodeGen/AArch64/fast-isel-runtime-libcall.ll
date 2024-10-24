; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -code-model=small -verify-machineinstrs < %s | FileCheck %s --check-prefix=SMALL
; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -code-model=large -verify-machineinstrs < %s | FileCheck %s --check-prefix=LARGE

define float @frem_f32(float %a, float %b) {
; SMALL-LABEL: frem_f32
; SMALL:       bl _fmodf
; LARGE-LABEL: frem_f32
; LARGE:       adrp  [[REG:x[0-9]+]], _fmodf@GOTPAGE
; LARGE:       ldr [[REG]], [[[REG]], _fmodf@GOTPAGEOFF]
; LARGE-NEXT:  blr [[REG]]
  %1 = frem float %a, %b
  ret float %1
}

define double @frem_f64(double %a, double %b) {
; SMALL-LABEL: frem_f64
; SMALL:       bl _fmod
; LARGE-LABEL: frem_f64
; LARGE:       adrp  [[REG:x[0-9]+]], _fmod@GOTPAGE
; LARGE:       ldr [[REG]], [[[REG]], _fmod@GOTPAGEOFF]
; LARGE-NEXT:  blr [[REG]]
  %1 = frem double %a, %b
  ret double %1
}

define float @sin_f32(float %a) {
; SMALL-LABEL: sin_f32
; SMALL:       bl _sinf
; LARGE-LABEL: sin_f32
; LARGE:       adrp  [[REG:x[0-9]+]], _sinf@GOTPAGE
; LARGE:       ldr [[REG]], [[[REG]], _sinf@GOTPAGEOFF]
; LARGE-NEXT:  blr [[REG]]
  %1 = call float @llvm.sin.f32(float %a)
  ret float %1
}

define double @sin_f64(double %a) {
; SMALL-LABEL: sin_f64
; SMALL:       bl _sin
; LARGE-LABEL: sin_f64
; LARGE:       adrp  [[REG:x[0-9]+]], _sin@GOTPAGE
; LARGE:       ldr [[REG]], [[[REG]], _sin@GOTPAGEOFF]
; LARGE-NEXT:  blr [[REG]]
  %1 = call double @llvm.sin.f64(double %a)
  ret double %1
}

define float @cos_f32(float %a) {
; SMALL-LABEL: cos_f32
; SMALL:       bl _cosf
; LARGE-LABEL: cos_f32
; LARGE:       adrp  [[REG:x[0-9]+]], _cosf@GOTPAGE
; LARGE:       ldr [[REG]], [[[REG]], _cosf@GOTPAGEOFF]
; LARGE-NEXT:  blr [[REG]]
  %1 = call float @llvm.cos.f32(float %a)
  ret float %1
}

define double @cos_f64(double %a) {
; SMALL-LABEL: cos_f64
; SMALL:       bl _cos
; LARGE-LABEL: cos_f64
; LARGE:       adrp  [[REG:x[0-9]+]], _cos@GOTPAGE
; LARGE:       ldr [[REG]], [[[REG]], _cos@GOTPAGEOFF]
; LARGE-NEXT:  blr [[REG]]
  %1 = call double @llvm.cos.f64(double %a)
  ret double %1
}

define float @tan_f32(float %a) {
; SMALL-LABEL: tan_f32
; SMALL:       bl _tanf
; LARGE-LABEL: tan_f32
; LARGE:       adrp  [[REG:x[0-9]+]], _tanf@GOTPAGE
; LARGE:       ldr [[REG]], [[[REG]], _tanf@GOTPAGEOFF]
; LARGE-NEXT:  blr [[REG]]
  %1 = call float @llvm.tan.f32(float %a)
  ret float %1
}

define double @tan_f64(double %a) {
; SMALL-LABEL: tan_f64
; SMALL:       bl _tan
; LARGE-LABEL: tan_f64
; LARGE:       adrp  [[REG:x[0-9]+]], _tan@GOTPAGE
; LARGE:       ldr [[REG]], [[[REG]], _tan@GOTPAGEOFF]
; LARGE-NEXT:  blr [[REG]]
  %1 = call double @llvm.tan.f64(double %a)
  ret double %1
}

define float @pow_f32(float %a, float %b) {
; SMALL-LABEL: pow_f32
; SMALL:       bl _powf
; LARGE-LABEL: pow_f32
; LARGE:       adrp  [[REG:x[0-9]+]], _powf@GOTPAGE
; LARGE:       ldr [[REG]], [[[REG]], _powf@GOTPAGEOFF]
; LARGE-NEXT:  blr [[REG]]
  %1 = call float @llvm.pow.f32(float %a, float %b)
  ret float %1
}

define double @pow_f64(double %a, double %b) {
; SMALL-LABEL: pow_f64
; SMALL:       bl _pow
; LARGE-LABEL: pow_f64
; LARGE:       adrp  [[REG:x[0-9]+]], _pow@GOTPAGE
; LARGE:       ldr [[REG]], [[[REG]], _pow@GOTPAGEOFF]
; LARGE-NEXT:  blr [[REG]]
  %1 = call double @llvm.pow.f64(double %a, double %b)
  ret double %1
}
declare float @llvm.sin.f32(float)
declare double @llvm.sin.f64(double)
declare float @llvm.cos.f32(float)
declare double @llvm.cos.f64(double)
declare float @llvm.tan.f32(float)
declare double @llvm.tan.f64(double)
declare float @llvm.pow.f32(float, float)
declare double @llvm.pow.f64(double, double)
