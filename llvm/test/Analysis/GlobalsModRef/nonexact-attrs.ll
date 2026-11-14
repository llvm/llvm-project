; RUN: opt < %s -passes='require<globals-aa>,aa-eval' -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; Functions without an exact definition (here: weak or noipa) may be replaced
; at link time, so their bodies can't be used. Their attributes still describe
; any replacement, though, so they can be used like those of a declaration.

@g = internal global i32 0

define weak void @weak_none() memory(none) {
  ret void
}

define weak void @weak_read() memory(read) {
  %v = load i32, ptr @g
  ret void
}

define void @noipa_none() noipa memory(none) {
  ret void
}

define void @caller_weak_none() {
  call void @weak_none()
  ret void
}

define void @caller_weak_read() {
  call void @weak_read()
  ret void
}

define void @caller_noipa_none() {
  call void @noipa_none()
  ret void
}

; An SCC containing a non-exact function.

define void @scc_exact() {
  call void @scc_weak_none()
  ret void
}

define weak void @scc_weak_none() memory(none) {
  call void @scc_exact()
  ret void
}

; CHECK-LABEL: Function: test_weak_none
; CHECK: NoModRef:  Ptr: i32* @g	<->  call void @caller_weak_none()
define i32 @test_weak_none() {
  store i32 1, ptr @g
  call void @caller_weak_none()
  %v = load i32, ptr @g
  ret i32 %v
}

; CHECK-LABEL: Function: test_weak_read
; CHECK: Just Ref:  Ptr: i32* @g	<->  call void @caller_weak_read()
define i32 @test_weak_read() {
  store i32 1, ptr @g
  call void @caller_weak_read()
  %v = load i32, ptr @g
  ret i32 %v
}

; CHECK-LABEL: Function: test_noipa_none
; CHECK: NoModRef:  Ptr: i32* @g	<->  call void @caller_noipa_none()
define i32 @test_noipa_none() {
  store i32 1, ptr @g
  call void @caller_noipa_none()
  %v = load i32, ptr @g
  ret i32 %v
}

; CHECK-LABEL: Function: test_scc
; CHECK: NoModRef:  Ptr: i32* @g	<->  call void @scc_exact()
define i32 @test_scc() {
  store i32 1, ptr @g
  call void @scc_exact()
  %v = load i32, ptr @g
  ret i32 %v
}
