; RUN: llc -mtriple=x86_64-apple-darwin10 -O0 < %s | FileCheck %s

; test that we print a label that we use. We had a bug where
; we would print the jump, but not the label because it was considered
; a fall through.

; CHECK:        jmp     LBB0_1
; CHECK: LBB0_1:

define void @foobar() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_zed()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  ret void

lpad:                                             ; preds = %entry
  %exn = landingpad {ptr, i32}
            cleanup
  unreachable
}

declare void @_zed() ssp align 2

declare i32 @__gxx_personality_v0(...)
