; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s 2>&1 | FileCheck %s

; CHECK: in function f
; CHECK-SAME: Cannot create Dot2AddHalf operation: No valid overloads for DXIL version 1.3

define noundef float @f(<2 x half> noundef %a, <2 x half> noundef %b, float %acc) {
entry:
  %ax = extractelement <2 x half> %a, i32 0
  %ay = extractelement <2 x half> %a, i32 1
  %bx = extractelement <2 x half> %b, i32 0
  %by = extractelement <2 x half> %b, i32 1
  
  %ret = call float @llvm.dx.dot2add(float %acc, half %ax, half %ay, half %bx, half %by)
  ret float %ret
}
