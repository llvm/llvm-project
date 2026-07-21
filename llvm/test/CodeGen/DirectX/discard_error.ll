; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation discard does not support no bool overload type

; CHECK: intrinsic argument 0 type expected i1, but got double
; CHECK: call void @llvm.dx.discard(double %p)
;
define void @discard_double(double noundef %p) {
entry:
  call void @llvm.dx.discard(double %p)
  ret void
}
