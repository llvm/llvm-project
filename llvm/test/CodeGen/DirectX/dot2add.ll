; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.4-compute %s | FileCheck %s

define noundef float @dot2add_simple(<2 x half> noundef %a, <2 x half> noundef %b, float %acc) {
entry:
  %ax = extractelement <2 x half> %a, i32 0
  %ay = extractelement <2 x half> %a, i32 1
  %bx = extractelement <2 x half> %b, i32 0
  %by = extractelement <2 x half> %b, i32 1

; CHECK: call float @dx.op.dot2AddHalf.f32(i32 162, float %acc, half %ax, half %ay, half %bx, half %by)
  %ret = call float @llvm.dx.dot2add(float %acc, half %ax, half %ay, half %bx, half %by)
  ret float %ret
}
