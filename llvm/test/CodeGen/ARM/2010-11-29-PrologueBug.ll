; RUN: llc < %s -mtriple=armv7-apple-ios   | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-ios | FileCheck %s
; rdar://8690640

define ptr @t(ptr %x) nounwind "frame-pointer"="all" {
entry:
; CHECK-LABEL: t:
; CHECK: push
; CHECK: mov r7, sp
; CHECK: bl _foo
; CHECK: bl _foo
; CHECK: bl _foo
; CHECK: pop {r7, pc}

  %0 = tail call ptr @foo(ptr %x) nounwind
  %1 = tail call ptr @foo(ptr %0) nounwind
  %2 = tail call ptr @foo(ptr %1) nounwind
  ret ptr %2
}

declare ptr @foo(ptr)
