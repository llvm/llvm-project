; RUN: llvm-link -S %s %p/Inputs/available_externally_over_decl.ll | FileCheck %s

declare void @f()

define available_externally void @g() {
  ret void
}

define ptr @main() {
  call void @g()
  ret ptr @f
}

; CHECK-DAG: define available_externally void @g() {
; CHECK-DAG: define available_externally void @f() {
