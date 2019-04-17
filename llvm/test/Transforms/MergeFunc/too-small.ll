; RUN: opt -S -mergefunc < %s | FileCheck %s

define void @foo(i32 %x) {
; CHECK-LABEL: @foo(
; CHECK-NOT: call
  ret void
}

define void @bar(i32 %x) {
; CHECK-LABEL: @bar(
; CHECK-NOT: call
  ret void
}

