; REQUIRES: asserts
; RUN: not --crash llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @f(token %A, token %B) {
entry:
  br label %bb

bb:
  %phi = phi token [ %A, %bb ], [ %B, %entry]
; CHECK: PHI nodes cannot have token type!
  br label %bb
}
