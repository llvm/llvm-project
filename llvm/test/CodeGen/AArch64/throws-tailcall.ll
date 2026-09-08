; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

; Herbception (throws): tail call optimization. When a throws function returns
; the result of another throws function directly, the backend should emit a
; tail call (b) rather than a call + ret sequence.

declare { i32, i1 } @tailcall_dest(i32) #0

define { i32, i1 } @tailcall(i32 %i) #0 {
; CHECK-LABEL: tailcall:
; CHECK:       // %bb.0:
; CHECK-NEXT:    b tailcall_dest
entry:
  %call = tail call { i32, i1 } @tailcall_dest(i32 %i)
  ret { i32, i1 } %call
}

attributes #0 = { nounwind throws }
