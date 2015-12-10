; RUN: llvm-link -S %s %p/Inputs/available_externally_over_decl.ll | FileCheck %s

declare void @f()

define void ()* @main() {
  ret void ()* @f
}

; CHECK: define available_externally void @f() {
