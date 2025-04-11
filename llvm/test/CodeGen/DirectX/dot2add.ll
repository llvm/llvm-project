; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

define noundef float @dot2add_simple(<2 x half> noundef %a, <2 x half> noundef %b, float %c) {
entry:
; CHECK: call float @dx.op.dot2AddHalf(i32 162, float %c, half %0, half %1, half %2, half %3)
  %ret = call float @llvm.dx.dot2add(<2 x half> %a, <2 x half> %b, float %c)
  ret float %ret
}
