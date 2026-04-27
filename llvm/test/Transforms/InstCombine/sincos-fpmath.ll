; RUN: opt -passes=instcombine -S < %s -mtriple=x86_64-apple-macosx10.9 | FileCheck %s

; Verify that when sin/cos libcalls are combined into llvm.sincos, the
; !fpmath metadata from the two original calls is merged via
; MDNode::getMostGenericFPMath and attached to the new sincos call.

declare float @llvm.sin.f32(float)
declare float @llvm.cos.f32(float)

; Both sin and cos carry !fpmath; the combined call should pick the tighter
; (smaller) accuracy bound, which is !0 (2.5 ULP).
define float @sincos_fpmath_metadata(float %x) {
; CHECK-LABEL: @sincos_fpmath_metadata(
; CHECK-NEXT:    [[SINCOS:%.*]] = call { float, float } @llvm.sincos.f32(float %x), !fpmath !0
; CHECK-NEXT:    [[S:%.*]] = extractvalue { float, float } [[SINCOS]], 0
; CHECK-NEXT:    [[C:%.*]] = extractvalue { float, float } [[SINCOS]], 1
; CHECK-NEXT:    [[RES:%.*]] = fadd float [[S]], [[C]]
; CHECK-NEXT:    ret float [[RES]]
;
  %s = call float @llvm.sin.f32(float %x), !fpmath !0
  %c = call float @llvm.cos.f32(float %x), !fpmath !1
  %res = fadd float %s, %c
  ret float %res
}

; If only one of the calls has fpmath, the combined call should have no fpmath.
define float @sincos_fpmath_one_unset(float %x) {
; CHECK-LABEL: @sincos_fpmath_one_unset(
; CHECK-NEXT:    [[SINCOS:%.*]] = call { float, float } @llvm.sincos.f32(float %x)
; CHECK-NEXT:    [[S:%.*]] = extractvalue { float, float } [[SINCOS]], 0
; CHECK-NEXT:    [[C:%.*]] = extractvalue { float, float } [[SINCOS]], 1
; CHECK-NEXT:    [[RES:%.*]] = fadd float [[S]], [[C]]
; CHECK-NEXT:    ret float [[RES]]
;
  %s = call float @llvm.sin.f32(float %x), !fpmath !0
  %c = call float @llvm.cos.f32(float %x)
  %res = fadd float %s, %c
  ret float %res
}

; CHECK: !0 = !{float 2.500000e+00}

!0 = !{float 2.5}
!1 = !{float 4.0}
