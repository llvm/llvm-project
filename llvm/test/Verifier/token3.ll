; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @f(token %A, token %B) {
entry:
  alloca token
; CHECK: invalid type for alloca
  ret void
}
