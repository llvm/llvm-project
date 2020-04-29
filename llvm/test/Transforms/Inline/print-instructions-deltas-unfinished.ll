; RUN: opt < %s -inline -debug-only=inline-cost -disable-output -print-instruction-deltas -inline-threshold=0 2>&1 | FileCheck %s

; CHECK: No analysis for the instruction
; CHECK:   ret void

declare void @callee1()

define void @bar() {
  call void @callee1()
  ret void
}

define void @foo() {
  call void @bar()
  ret void
}
