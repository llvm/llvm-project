; REQUIRES: amdgpu-registered-target
; RUN: opt -S -passes=speculative-execution -mtriple=amdgcn--  \
; RUN: -spec-exec-only-if-divergent-target \
; RUN:   -spec-exec-max-speculation-cost 4 -spec-exec-max-not-hoisted 3 \
; RUN:   %s | FileCheck %s

; Hoist in if-then pattern.
define void @skip_single_lane_ifThen() #0 {
; CHECK-LABEL: @skip_single_lane_ifThen(
; CHECK: br i1 true

br i1 true, label %a, label %b
; CHECK: a:
; CHECK: %x = add i32 2, 3
a:
  %x = add i32 2, 3
; CHECK: br label
  br label %b
; CHECK: b:
b:
; CHECK: ret void
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1" }
