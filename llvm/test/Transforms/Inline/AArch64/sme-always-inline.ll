; REQUIRES: aarch64-registered-target
; RUN: opt -passes=always-inline -S < %s | FileCheck %s

target triple = "aarch64"

define internal float @callee(float %v) #0 {
; CHECK-LABEL: define internal float @callee(
; CHECK-SAME: float [[V:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:    [[RES:%.*]] = tail call float @llvm.sin.f32(float [[V]])
; CHECK-NEXT:    ret float [[RES]]
;
  %res = tail call float @llvm.sin.f32(float %v)
  ret float %res
}

define float @caller(float %v) #1 {
; CHECK-LABEL: define float @caller(
; CHECK-SAME: float [[V:%.*]]) #[[ATTR1:[0-9]+]] {
; CHECK-NEXT:    [[RES:%.*]] = call float @callee(float [[V]])
; CHECK-NEXT:    ret float [[RES]]
;
  %res = call float @callee(float %v)
  ret float %res
}

attributes #0 = { "target-features"="+sme2" nounwind alwaysinline }
attributes #1 = { "target-features"="+sme2" nounwind "aarch64_pstate_sm_enabled" }
