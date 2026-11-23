; RUN: llc < %s  -tailcallopt -mtriple=i686-pc-linux-gnu -relocation-model=pic | FileCheck %s

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
; CHECK-LABEL: tailcall_hidden:
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
; CHECK-LABEL: tailcall_internal:
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
; CHECK-LABEL: tailcall_default:
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
; CHECK-LABEL: tailcall_default_implicit:
; CHECK: calll tailcallee_default_implicit@PLT

define void @tailcall_external() {
  tail call void @external()
  ret void
}
; CHECK-LABEL: tailcall_external:
; CHECK: calll external@PLT

define void @musttail_external() {
  musttail call void @external()
  ret void
}
; CHECK-LABEL: musttail_external:
; CHECK: movl external@GOT
; CHECK: jmpl

; This test uses guaranteed TCO so these will be tail calls, despite the early
; binding issues.

define protected fastcc i32 @tailcallee_protected_fastcc(i32 %a1, i32 %a2, i32 %a3, i32 %a4) {
entry:
	ret i32 %a3
}

define fastcc i32 @tailcaller_protected_fastcc(i32 %in1, i32 %in2) {
entry:
	%tmp11 = tail call fastcc i32 @tailcallee_protected_fastcc( i32 %in1, i32 %in2, i32 %in1, i32 %in2 )		; <i32> [#uses=1]
	ret i32 %tmp11
; CHECK-LABEL: tailcaller_protected_fastcc:
; CHECK: jmp tailcallee_protected_fastcc
}

define fastcc i32 @tailcallee_default_fastcc(i32 %a1, i32 %a2, i32 %a3, i32 %a4) {
entry:
	ret i32 %a3
}

define fastcc i32 @tailcaller_default_fastcc(i32 %in1, i32 %in2) {
entry:
	%tmp11 = tail call fastcc i32 @tailcallee_default_fastcc( i32 %in1, i32 %in2, i32 %in1, i32 %in2 )		; <i32> [#uses=1]
	ret i32 %tmp11
; CHECK-LABEL: tailcaller_default_fastcc:
; CHECK: movl tailcallee_default_fastcc@GOT
; CHECK: jmpl
}

define i32 @tailcall_indirect(ptr %fp) {
  %rv = tail call i32 () %fp()
  ret i32 %rv
; CHECK-LABEL: tailcall_indirect:
; CHECK: jmpl {{.*}} # TAILCALL
}
