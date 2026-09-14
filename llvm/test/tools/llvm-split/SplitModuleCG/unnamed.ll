; RUN: llvm-split -enable-call-graph-split-module=true -j2 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s

; Test that an unnamed internal function (@0) is given a stable name and
; .llvm.<suffix> suffix after promotion.

; CHECK0-DAG: declare hidden void @__llvm_unnamed.llvm.{{[0-9a-f]+}}()
; CHECK0-DAG: define hidden void @__llvm_unnamed.1.llvm.{{[0-9a-f]+}}()

; CHECK1-DAG: define hidden void @__llvm_unnamed.llvm.{{[0-9a-f]+}}()
; CHECK1-DAG: declare hidden void @__llvm_unnamed.1.llvm.{{[0-9a-f]+}}()

define internal void @0() {
  ret void
}

define internal void @1() {
  ret void
}
