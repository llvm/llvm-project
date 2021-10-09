; RUN: llvm-link %p/Inputs/fixed-vector-type-construction.ll %s -S -o - | FileCheck %s
%t = type {i32, float}
; CHECK: define void @foo(<4 x
; CHECK: define void @bar(<vscale x 4 x
define void @bar(<vscale x 4 x %t*> %x) {
  ret void
}
