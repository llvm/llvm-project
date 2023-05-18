; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as -o %t/1.bc %s
; RUN: llvm-as -o %t/2.bc %S/Inputs/list-symbols.ll
; RUN: llvm-lto -list-symbols-only %t/1.bc %t/2.bc | FileCheck %s
; REQUIRES: default_triple

; CHECK-LABEL: 1.bc:
; CHECK-DAG: foo
; CHECK-DAG: glob
; CHECK-LABEL: 2.bc:
; CHECK-DAG: glob
; CHECK-DAG: bar

@glob = global i32 0
define void @foo() {
  ret void
}
