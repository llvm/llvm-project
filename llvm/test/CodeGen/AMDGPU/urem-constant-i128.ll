; RUN: llc -global-isel=0 -mtriple=amdgpu6.00 < %s | FileCheck %s --implicit-check-not=s_swappc_b64
; RUN: llc -global-isel=0 -mtriple=amdgpu11.00 < %s | FileCheck %s --implicit-check-not=s_swappc_b64

; Check that AMDGPU legalizes constant i128 UREM without emitting a libcall.

define i128 @urem_i128_65(i128 %src) {
; CHECK-LABEL: urem_i128_65:
; CHECK:       s_setpc_b64
  %result = urem i128 %src, 65
  ret i128 %result
}

define i128 @urem_i128_65_optsize(i128 %src) #0 {
; CHECK-LABEL: urem_i128_65_optsize:
; CHECK:       s_setpc_b64
  %result = urem i128 %src, 65
  ret i128 %result
}

define i128 @urem_i128_66(i128 %src) {
; CHECK-LABEL: urem_i128_66:
; CHECK:       s_setpc_b64
  %result = urem i128 %src, 66
  ret i128 %result
}

define i128 @urem_i128_67(i128 %src) {
; CHECK-LABEL: urem_i128_67:
; CHECK:       s_setpc_b64
  %result = urem i128 %src, 67
  ret i128 %result
}

define i128 @urem_i128_68(i128 %src) {
; CHECK-LABEL: urem_i128_68:
; CHECK:       s_setpc_b64
  %result = urem i128 %src, 68
  ret i128 %result
}

define i128 @urem_i128_127(i128 %src) {
; CHECK-LABEL: urem_i128_127:
; CHECK:       s_setpc_b64
  %result = urem i128 %src, 127
  ret i128 %result
}

attributes #0 = { optsize }
