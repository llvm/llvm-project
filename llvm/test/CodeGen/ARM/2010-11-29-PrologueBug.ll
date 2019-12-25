; RUN: llc < %s -mtriple=armv7-apple-ios   | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-ios | FileCheck %s
; rdar://8690640

define i32* @t(i32* %x) nounwind "frame-pointer"="all" {
entry:
; CHECK-LABEL: t:
; CHECK: push
; CHECK: mov r7, sp
; CHECK: bl _foo
; CHECK: bl _foo
; CHECK: bl _foo
; CHECK: pop {r7, pc}

  %0 = tail call i32* @foo(i32* %x) nounwind
  %1 = tail call i32* @foo(i32* %0) nounwind
  %2 = tail call i32* @foo(i32* %1) nounwind
  ret i32* %2
}

declare i32* @foo(i32*)
