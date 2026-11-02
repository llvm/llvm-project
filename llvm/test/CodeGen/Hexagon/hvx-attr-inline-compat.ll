; A "hexagon_hvx" callee must not be inlined into a caller without the
; attribute, in either inliner, and even when marked alwaysinline. Enforced in
; HexagonTTIImpl::areInlineCompatible rather than through an Attributes.td
; CompatRule, because the generic attribute check is skipped for alwaysinline
; callees.

; RUN: opt -mtriple=hexagon -passes=always-inline -S < %s | FileCheck %s
; RUN: opt -mtriple=hexagon -passes=inline -S < %s | FileCheck %s

define internal void @hvx_callee(ptr %p) alwaysinline #0 {
  store i32 1, ptr %p, align 4
  ret void
}

define internal void @plain_callee(ptr %p) alwaysinline #1 {
  store i32 2, ptr %p, align 4
  ret void
}

; The attribute differs, so the call survives.
; CHECK-LABEL: define void @plain_caller(
; CHECK: call void @hvx_callee(
define void @plain_caller(ptr %p) #1 {
  call void @hvx_callee(ptr %p)
  ret void
}

; Both sides declared for HVX, so this inlines.
; CHECK-LABEL: define void @hvx_caller(
; CHECK-NOT: call void @hvx_callee(
define void @hvx_caller(ptr %p) #0 {
  call void @hvx_callee(ptr %p)
  ret void
}

; The rule is one-directional: a plain callee still inlines into an HVX caller.
; CHECK-LABEL: define void @hvx_caller_plain_callee(
; CHECK-NOT: call void @plain_callee(
define void @hvx_caller_plain_callee(ptr %p) #0 {
  call void @plain_callee(ptr %p)
  ret void
}

attributes #0 = { nounwind "hexagon_hvx" "target-features"="+hvxv68,+hvx-length128b" }
attributes #1 = { nounwind "target-features"="+hvxv68,+hvx-length128b" }
