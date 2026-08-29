; RUN: llvm-split -enable-call-graph-split-module=true -j1 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s

; Test that -j1 places all functions into a single partition.

; CHECK0: define void @foo()
; CHECK0: define void @bar()

define void @foo() {
  call void @bar()
  ret void
}

define void @bar() {
  ret void
}
