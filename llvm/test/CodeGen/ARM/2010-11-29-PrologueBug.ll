; RUN: llc < %s -mtriple=armv7-apple-ios   | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB2
; rdar://8690640

define i32* @t(i32* %x) nounwind {
entry:
; ARM-LABEL: t:
; ARM: push
; ARM: mov r7, sp
; ARM: bl _foo
; ARM: bl _foo
; ARM: bl _foo
; ARM: pop {r7, pc}

; THUMB2-LABEL: t:
; THUMB2: push
; THUMB2: mov r7, sp
; THUMB2: blx _foo
; THUMB2: blx _foo
; THUMB2: blx _foo
; THUMB2: pop
  %0 = tail call i32* @foo(i32* %x) nounwind
  %1 = tail call i32* @foo(i32* %0) nounwind
  %2 = tail call i32* @foo(i32* %1) nounwind
  ret i32* %2
}

declare i32* @foo(i32*)
