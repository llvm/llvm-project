; Inlining must not mix HMX and non-HMX bodies in either direction, and the
; check has to hold for alwaysinline callees, which is why this is enforced in
; TTI rather than through an Attributes.td CompatRule.

; RUN: opt -mtriple=hexagon -passes=always-inline -S < %s | FileCheck %s

define internal void @hvx_helper(ptr %p) alwaysinline #1 {
  store i32 1, ptr %p, align 4
  ret void
}

define internal void @hmx_helper(ptr %p) alwaysinline #0 {
  store i32 2, ptr %p, align 4
  ret void
}

; CHECK-LABEL: define void @hmx_caller(
; CHECK:         call void @hvx_helper(
define void @hmx_caller(ptr %p) #0 {
  call void @hvx_helper(ptr %p)
  ret void
}

; CHECK-LABEL: define void @hvx_caller(
; CHECK:         call void @hmx_helper(
define void @hvx_caller(ptr %p) #1 {
  call void @hmx_helper(ptr %p)
  ret void
}

; Matching attributes still inline, so the checks above are not passing because
; alwaysinline is broken.
; CHECK-LABEL: define void @hmx_caller_same(
; CHECK-NOT:     call void @hmx_helper(
define void @hmx_caller_same(ptr %p) #0 {
  call void @hmx_helper(ptr %p)
  ret void
}

attributes #0 = { nounwind "hexagon_hmx" "target-features"="+hvxv68,+hvx-length128b" }
attributes #1 = { nounwind "target-features"="+hvxv68,+hvx-length128b" }
