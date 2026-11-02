; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

; Reject llvm.mask.beforefirst with non-mask vector arguments.

define <4 x i8> @v4i8(<4 x i8> %m) {
; CHECK: mask.beforefirst element type must be i1
  %res = call <4 x i8> @llvm.mask.beforefirst(<4 x i8> %m)
  ret <4 x i8> %res
}
