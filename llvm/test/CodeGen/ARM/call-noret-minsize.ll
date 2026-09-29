; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=cortex-a8   | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=swift       | FileCheck %s -check-prefix=SWIFT
; rdar://12348580

define void @t1() noreturn minsize nounwind ssp {
entry:
; ARM-LABEL: t1:
; ARM: bl _bar

; SWIFT-LABEL: t1:
; SWIFT: bl _bar
  tail call void @bar() noreturn nounwind
  unreachable
}

define void @t2() noreturn minsize nounwind ssp {
entry:
; ARM-LABEL: t2:
; ARM: bl _t1

; SWIFT-LABEL: t2:
; SWIFT: bl _t1
  tail call void @t1() noreturn nounwind
  unreachable
}

declare void @bar() noreturn
