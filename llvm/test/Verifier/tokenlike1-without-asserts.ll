; REQUIRES: !asserts
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @f(target("dx.RawBuffer", half, 1, 0) %A, target("dx.RawBuffer", half, 1, 0) %B) {
entry:
  br label %bb

bb:
  %phi = phi target("dx.RawBuffer", half, 1, 0) [ %A, %bb ], [ %B, %entry]
; CHECK: PHI nodes cannot have token type!
  br label %bb
}
