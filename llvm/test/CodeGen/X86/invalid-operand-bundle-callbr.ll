; RUN: not llc -mtriple=x86_64-unknown-linux-gnu < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: cannot lower callbrs with arbitrary operand bundles: foo

define void @f(i32 %arg) {
  callbr void asm "", ""() [ "foo"(i32 %arg) ]
    to label %cont []

cont:
  ret void
}
