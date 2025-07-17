; RUN: not llc -mtriple=x86_64-unknown-linux-gnu < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: cannot lower calls with arbitrary operand bundles: foo

declare void @g()

define void @f(i32 %arg) {
  call void @g() [ "foo"(i32 %arg) ]
  ret void
}
