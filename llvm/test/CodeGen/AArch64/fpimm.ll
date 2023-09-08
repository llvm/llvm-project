; RUN: llc -mtriple=aarch64-linux-gnu                                                  -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -code-model=large                             -verify-machineinstrs < %s | FileCheck %s --check-prefixes=LARGE
; RUN: llc -mtriple=aarch64    -code-model=tiny                              -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel                           -verify-machineinstrs < %s | FileCheck %s

@varf32 = global float 0.0
@varf64 = global double 0.0

define void @check_float() {
; CHECK-LABEL: check_float:

  %val = load float, ptr @varf32
  %newval1 = fadd float %val, 8.5
  store volatile float %newval1, ptr @varf32
; CHECK-DAG: fmov {{s[0-9]+}}, #8.5

  %newval2 = fadd float %val, 128.0
  store volatile float %newval2, ptr @varf32
; CHECK-DAG: movi [[REG:v[0-9s]+]].2s, #67, lsl #24

; CHECK: ret
  ret void
}

define void @check_double() {
; CHECK-LABEL: check_double:

  %val = load double, ptr @varf64
  %newval1 = fadd double %val, 8.5
  store volatile double %newval1, ptr @varf64
; CHECK-DAG: fmov {{d[0-9]+}}, #8.5

  %newval2 = fadd double %val, 128.0
  store volatile double %newval2, ptr @varf64
; CHECK-DAG: mov [[X128:x[0-9]+]], #4638707616191610880
; CHECK-DAG: fmov {{d[0-9]+}}, [[X128]]

; 64-bit ORR followed by MOVK.
; CHECK-DAG: mov  [[XFP0:x[0-9]+]], #1082331758844
; CHECK-DAG: movk [[XFP0]], #64764, lsl #16
; CHECk-DAG: fmov {{d[0-9]+}}, [[XFP0]]
  %newval3 = fadd double %val, 0xFCFCFC00FC
  store volatile double %newval3, ptr @varf64

; CHECK: ret
  ret void
}

; CHECK-LABEL: check_float2
; CHECK:       mov [[REG:w[0-9]+]], #4059
; CHECK-NEXT:  movk [[REG]], #16457, lsl #16
; CHECK-NEXT:  fmov {{s[0-9]+}}, [[REG]]
define float @check_float2() {
  ret float 3.14159274101257324218750
}

; LARGE-LABEL: check_double2
; LARGE:       mov [[REG:x[0-9]+]], #11544
; LARGE-NEXT:  movk [[REG]], #21572, lsl #16
; LARGE-NEXT:  movk [[REG]], #8699, lsl #32
; LARGE-NEXT:  movk [[REG]], #16393, lsl #48
; LARGE-NEXT:  fmov {{d[0-9]+}}, [[REG]]
define double @check_double2() {
  ret double 3.1415926535897931159979634685441851615905761718750
}
