; RUN: opt -passes=always-inline -S < %s | FileCheck %s

target triple = "aarch64"

define internal float @callee(float %v) "target-features"="+sme2" alwaysinline {
; CHECK-NOT: define internal float @callee
  %res = tail call float @llvm.sin.f32(float %v)
  ret float %res
}

; Test that the body of @callee is inlined due to it's `alwaysinline` attribute,
; despite having mismatching streaming attributes.
define float @caller(float %v) "target-features"="+sme2" "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: define float @caller(
; CHECK-SAME: float [[V:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:    [[RES:%.*]] = call float @llvm.sin.f32(float [[V]])
; CHECK-NEXT:    ret float [[RES]]
;
  %res = call float @callee(float %v)
  ret float %res
}
