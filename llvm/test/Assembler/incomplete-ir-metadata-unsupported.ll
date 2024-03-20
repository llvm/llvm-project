; RUN: not llvm-as -allow-incomplete-ir < %s 2>&1 | FileCheck %s

; CHECK: error: use of undefined metadata '!1'
define void @test(ptr %p) {
  %v = load i8, ptr %p, !noalias !0
  ret void
}

!0 = !{!1}
