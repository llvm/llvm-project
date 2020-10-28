; RUN: llc < %s -mtriple=i686-pc-linux-gnu -relocation-model=pic | FileCheck %s

; While many of these could be tail called, we don't do it because it forces
; early binding.

declare void @external()

define hidden void @tailcallee_hidden() {
entry:
  ret void
}

define void @tailcall_hidden() {
entry:
  tail call void @tailcallee_hidden()
  ret void
}
; CHECK: tailcall_hidden:
; CHECK: jmp tailcallee_hidden

define internal void @tailcallee_internal() {
entry:
  ret void
}

define void @tailcall_internal() {
entry:
  tail call void @tailcallee_internal()
  ret void
}
; CHECK: tailcall_internal:
; CHECK: jmp tailcallee_internal

define default void @tailcallee_default() {
entry:
  ret void
}

define void @tailcall_default() {
entry:
  tail call void @tailcallee_default()
  ret void
}
; CHECK: tailcall_default:
; CHECK: calll tailcallee_default@PLT

define void @tailcallee_default_implicit() {
entry:
  ret void
}

define void @tailcall_default_implicit() {
entry:
  tail call void @tailcallee_default_implicit()
  ret void
}
; CHECK: tailcall_default_implicit:
; CHECK: calll tailcallee_default_implicit@PLT

define void @tailcall_external() {
  tail call void @external()
  ret void
}
; CHECK: tailcall_external:
; CHECK: calll external@PLT

define void @musttail_external() {
  musttail call void @external()
  ret void
}
; CHECK: musttail_external:
; CHECK: movl external@GOT
; CHECK: jmpl
