; RUN: llc -mtriple=aarch64-linux-gnu                                                  -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -code-model=large                             -verify-machineinstrs < %s | FileCheck %s --check-prefix=LARGE
; RUN: llc -mtriple=aarch64-apple-darwin -code-model=large -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s --check-prefix=LARGE

@varf32 = global float 0.0
@varf64 = global double 0.0

define void @check_float() {
; CHECK-LABEL: check_float:

  %val = load float, float* @varf32
  %newval1 = fadd float %val, 8.5
  store volatile float %newval1, float* @varf32
; CHECK-DAG: fmov [[EIGHT5:s[0-9]+]], #8.5

  %newval2 = fadd float %val, 128.0
  store volatile float %newval2, float* @varf32
; CHECK-DAG: ldr [[HARD:s[0-9]+]], [{{x[0-9]+}}, {{#?}}:lo12:.LCPI0_0

; CHECK: ret
  ret void
}

define void @check_double() {
; CHECK-LABEL: check_double:

  %val = load double, double* @varf64
  %newval1 = fadd double %val, 8.5
  store volatile double %newval1, double* @varf64
; CHECK-DAG: fmov {{d[0-9]+}}, #8.5

  %newval2 = fadd double %val, 128.0
  store volatile double %newval2, double* @varf64
; CHECK-DAG: ldr {{d[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.LCPI1_0

; CHECK: ret
  ret void
}

; LARGE-LABEL: check_float2
; LARGE:       mov [[REG:w[0-9]+]], #1078525952
; LARGE-NEXT:  movk [[REG]], #4059
; LARGE-NEXT:  fmov s0, [[REG]]
define float @check_float2() {
  ret float 3.14159274101257324218750
}

; LARGE-LABEL: check_double2
; LARGE:       mov [[REG:x[0-9]+]], #4614219293217783808
; LARGE-NEXT:  movk [[REG]], #8699, lsl #32
; LARGE-NEXT:  movk [[REG]], #21572, lsl #16
; LARGE-NEXT:  movk [[REG]], #11544
; LARGE-NEXT:  fmov d0, [[REG]]
define double @check_double2() {
  ret double 3.1415926535897931159979634685441851615905761718750
}
