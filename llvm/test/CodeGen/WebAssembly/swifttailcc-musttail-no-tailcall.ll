; RUN: not llc < %s -mtriple=wasm32-unknown-unknown -verify-machineinstrs 2>&1 | FileCheck %s

; musttail with swifttailcc requires the +tail-call feature.
; CHECK: error:
; CHECK-SAME: WebAssembly 'tail-call' feature not enabled

define swifttailcc void @musttail_no_tailcall(ptr swiftasync %ctx) {
  musttail call swifttailcc void @musttail_no_tailcall(ptr swiftasync %ctx)
  ret void
}
