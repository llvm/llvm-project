; RUN: not llc < %s -mtriple=wasm32-unknown-unknown -mattr=+tail-call 2>&1 | FileCheck %s

; A byval argument is copied into the caller's frame, which a tail call
; releases before the callee reads it.
; CHECK: error:
; CHECK-SAME: WebAssembly does not support tail calling with byval arguments

declare i32 @quux(ptr byval(i32))

define i32 @musttail_byval(ptr byval(i32) %x) {
  %v = musttail call i32 @quux(ptr byval(i32) %x)
  ret i32 %v
}
