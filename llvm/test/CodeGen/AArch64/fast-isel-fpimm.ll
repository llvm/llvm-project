; RUN: llc -mtriple=aarch64-linux-gnu -code-model=large -fast-isel -fast-isel-abort=1  -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -code-model=large -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: check_float2
; CHECK:       mov [[REG:w[0-9]+]], #4059
; CHECK:       movk [[REG]], #16457, lsl #16
; CHECK-NEXT:  fmov {{s[0-9]+}}, [[REG]]
define float @check_float2() {
  ret float 3.14159274101257324218750
}

; CHECK-LABEL: check_double2
; CHECK:       mov [[REG:x[0-9]+]], #11544
; CHECK-NEXT:  movk [[REG]], #21572, lsl #16
; CHECK-NEXT:  movk [[REG]], #8699, lsl #32
; CHECK-NEXT:  movk [[REG]], #16393, lsl #48
; LARGE-NEXT:  fmov {{d[0-9]+}}, [[REG]]
define double @check_double2() {
  ret double 3.1415926535897931159979634685441851615905761718750
}
