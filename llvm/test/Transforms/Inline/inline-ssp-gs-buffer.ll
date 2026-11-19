; RUN: opt -passes=inline %s -S | FileCheck %s
; Inlining must not leave a caller applying MSVC's narrower /GS heuristic to
; code that was compiled with the GCC-compatible strong heuristic.

declare void @sink(ptr)

define internal void @callee_strong() sspstrong {
  %a = alloca [4 x i8]
  call void @sink(ptr %a)
  ret void
}

define internal void @callee_gs() sspstrong "stack-protector-gs-buffer"="true" {
  %a = alloca [64 x i8]
  call void @sink(ptr %a)
  ret void
}

define internal void @callee_nossp() {
  ret void
}

; Inlining a genuinely-strong callee drops the caller's /GS marker.
; CHECK: define void @strong_into_gs() #[[STRONG:[0-9]+]] {
define void @strong_into_gs() sspstrong "stack-protector-gs-buffer"="true" {
  call void @callee_strong()
  ret void
}

; Inlining another /GS function leaves the marker in place.
; CHECK: define void @gs_into_gs() #[[GS:[0-9]+]] {
define void @gs_into_gs() sspstrong "stack-protector-gs-buffer"="true" {
  call void @callee_gs()
  ret void
}

; A callee with no stack protector attribute at all must not disturb the
; marker, since it does not ask for any protection of its own.
; CHECK: define void @nossp_into_gs() #[[GS]] {
define void @nossp_into_gs() sspstrong "stack-protector-gs-buffer"="true" {
  call void @callee_nossp()
  ret void
}

; CHECK: attributes #[[STRONG]] = { sspstrong }
; CHECK: attributes #[[GS]] = { sspstrong "stack-protector-gs-buffer"="true" }
