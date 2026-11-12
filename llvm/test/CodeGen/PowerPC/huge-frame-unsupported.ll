; REQUIRES: asserts
; RUN: not --crash llc -verify-machineinstrs -mtriple=powerpc-unknown-unknown < %s \
; RUN:   2>&1 | FileCheck %s

declare void @bar(ptr)

define void @foo(i8 %x) {
; CHECK: Unhandled stack size
entry:
  %a = alloca i8, i64 4294967296, align 16
  store volatile i8 %x, ptr %a
  ret void
}
