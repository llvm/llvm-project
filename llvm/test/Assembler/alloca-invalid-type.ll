; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: invalid type for alloca

define void @test() {
entry:
  alloca metadata !{null}
  ret void
}
