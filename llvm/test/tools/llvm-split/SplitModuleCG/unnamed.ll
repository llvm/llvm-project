; RUN: llvm-split -enable-call-graph-split-module=true -j2 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s

; Test that an unnamed internal function (@0) is given a stable name and
; .llvm.<suffix> suffix after promotion.

; CHECK0-DAG: {{define hidden void @__llvmsplit_unnamed\.llvm\.}}

define internal void @0() {
  ret void
}
