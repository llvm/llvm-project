; RUN: llc -global-isel=0 -mtriple=amdgpu6.00 < %s | FileCheck %s --implicit-check-not=s_swappc_b64
; RUN: llc -global-isel=0 -mtriple=amdgpu11.00 < %s | FileCheck %s --implicit-check-not=s_swappc_b64

; The generated remainder and funnel-shift expansions are long and subject to
; frequent changes. Check that SelectionDAG can lower variable funnel shifts
; whose irregular types promote to i128 without requiring an i128 urem libcall.

declare i65 @llvm.fshl.i65(i65, i65, i65)
declare i66 @llvm.fshl.i66(i66, i66, i66)
declare i67 @llvm.fshl.i67(i67, i67, i67)
declare i68 @llvm.fshl.i68(i68, i68, i68)
declare i127 @llvm.fshl.i127(i127, i127, i127)

define i65 @fshl_i65(i65 %amt) {
; CHECK-LABEL: fshl_i65:
; CHECK:       s_setpc_b64
  %result = call i65 @llvm.fshl.i65(i65 1, i65 0, i65 %amt)
  ret i65 %result
}

define i66 @fshl_i66(i66 %amt) {
; CHECK-LABEL: fshl_i66:
; CHECK:       s_setpc_b64
  %result = call i66 @llvm.fshl.i66(i66 1, i66 0, i66 %amt)
  ret i66 %result
}

define i67 @fshl_i67(i67 %amt) {
; CHECK-LABEL: fshl_i67:
; CHECK:       s_setpc_b64
  %result = call i67 @llvm.fshl.i67(i67 1, i67 0, i67 %amt)
  ret i67 %result
}

define i68 @fshl_i68(i68 %amt) {
; CHECK-LABEL: fshl_i68:
; CHECK:       s_setpc_b64
  %result = call i68 @llvm.fshl.i68(i68 1, i68 0, i68 %amt)
  ret i68 %result
}

define i127 @fshl_i127(i127 %amt) {
; CHECK-LABEL: fshl_i127:
; CHECK:       s_setpc_b64
  %result = call i127 @llvm.fshl.i127(i127 1, i127 0, i127 %amt)
  ret i127 %result
}
