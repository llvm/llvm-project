; RUN: llc < %s -mtriple=wasm32-unknown-unknown -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Without +tail-call, advisory `tail` calls silently fall back to regular calls.

define swifttailcc void @basic_no_tailcall(ptr swiftasync %ctx) {
; CHECK-LABEL: basic_no_tailcall:
; CHECK:         .functype basic_no_tailcall (i32, i32, i32) -> ()
; CHECK:         return
  ret void
}

; Advisory tail call without +tail-call: falls back to regular call.
define swifttailcc void @tail_no_tailcall(ptr swiftasync %ctx) {
; CHECK-LABEL: tail_no_tailcall:
; CHECK-NOT:     return_call
; CHECK:         call tail_no_tailcall
  tail call swifttailcc void @tail_no_tailcall(ptr swiftasync %ctx)
  ret void
}
