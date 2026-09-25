; Test that we get an appropriate error when parsing a summary that does
; not have value info associated with a function definition.

; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: expected function definition foo to have an associated value info.

define void @foo() {
  ret void
}
^1 = gv: (name: "foo")
