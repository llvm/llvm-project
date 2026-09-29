; Test that the InlineAdvisor, upon being cleared, is re-created correctly.
; RUN: opt -S -passes="default<O1>,cgscc(inline)" < %s | FileCheck %s

define double @foo() local_unnamed_addr {
entry:
  ret double undef
}

; CHECK: @foo
