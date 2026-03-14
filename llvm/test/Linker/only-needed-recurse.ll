; RUN: llvm-link -S -only-needed %s %p/Inputs/only-needed-recurse.ll | FileCheck %s

declare void @f2()

define void @f1() {
  call void @f2()
  ret void
}

; CHECK: define void @f3

