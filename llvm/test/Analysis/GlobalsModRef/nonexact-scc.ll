; RUN: opt < %s -passes='require<globals-aa>,aa-eval' -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; Each SCC below contains one exact and one non-exact (weak) definition. The
; weak definition may be replaced at link time by one that calls @set_g, so
; GlobalsAA must not use the bodies in the SCC to conclude that calls into it
; don't access @g, regardless of which function comes first in the SCC.

@g = internal global i32 0

define void @set_g() {
  store i32 1, ptr @g
  ret void
}

; Exact function defined first.

define void @exact1() {
  call void @weak1()
  ret void
}

define weak void @weak1() {
  call void @exact1()
  ret void
}

; Weak function defined first.

define weak void @weak2() {
  call void @exact2()
  ret void
}

define void @exact2() {
  call void @weak2()
  ret void
}

; CHECK-LABEL: Function: test1
; CHECK: Both ModRef:  Ptr: i32* @g	<->  call void @exact1()
define i32 @test1() {
  store i32 1, ptr @g
  call void @exact1()
  %v = load i32, ptr @g
  ret i32 %v
}

; CHECK-LABEL: Function: test2
; CHECK: Both ModRef:  Ptr: i32* @g	<->  call void @exact2()
define i32 @test2() {
  store i32 1, ptr @g
  call void @exact2()
  %v = load i32, ptr @g
  ret i32 %v
}
