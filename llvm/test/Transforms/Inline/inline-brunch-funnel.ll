; Test that inliner skips @llvm.icall.branch.funnel
; RUN: opt < %s -passes=inline -S | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.icall.branch.funnel(...)

; CHECK-LABEL: define void @fn_musttail(
define void @fn_musttail() {
  call void (...) @bf_musttail()
  ; CHECK: call void (...) @bf_musttail()
  ret void
}

; CHECK-LABEL: define internal void @bf_musttail(
define internal void @bf_musttail(...) {
  musttail call void (...) @llvm.icall.branch.funnel(...)
  ; CHECK: musttail call void (...) @llvm.icall.branch.funnel(...)
  ret void
}

; CHECK-LABEL: define void @fn_musttail_always(
define void @fn_musttail_always() {
  call void (...) @bf_musttail_always()
  ; CHECK: call void (...) @bf_musttail_always()
  ret void
}

; CHECK-LABEL: define internal void @bf_musttail_always(
define internal void @bf_musttail_always(...) alwaysinline {
  musttail call void (...) @llvm.icall.branch.funnel(...)
  ; CHECK: musttail call void (...) @llvm.icall.branch.funnel(...)
  ret void
}
