; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=i686-unknown-linux-gnu < %s | FileCheck %s --check-prefix=CHECK32

; Herbception (throws): tail call optimization. When a throws function returns
; the result of another throws function directly, the backend should emit a
; tail call (jmp) rather than a call + ret sequence.

declare { i32, i1 } @tailcall_dest(i32) #0

define { i32, i1 } @tailcall(i32 %i) #0 {
; CHECK-LABEL: tailcall:
; CHECK:       # %bb.0:
; CHECK-NEXT:    jmp tailcall_dest@PLT # TAILCALL
; CHECK32-LABEL: tailcall:
; CHECK32:       # %bb.0:
; CHECK32-NEXT:    jmp tailcall_dest@PLT # TAILCALL
entry:
  %call = tail call { i32, i1 } @tailcall_dest(i32 %i)
  ret { i32, i1 } %call
}

attributes #0 = { nounwind throws }
