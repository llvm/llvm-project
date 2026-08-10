; RUN: llvm-cat -o - %s %S/Inputs/multi-module.ll | llvm-nm - | FileCheck %s

; CHECK: T f1
; CHECK: T f2

define void @f1() {
  ret void
}
