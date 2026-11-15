; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; The midpoint between these adjacent bfloat values is finite in float.
; Adding the endpoints before halving would overflow.
define i1 @above_bfloat(float %x) {
; CHECK-LABEL: @above_bfloat(
; CHECK: fcmp ogt float %x, f0x7F008000
  %narrow = fptrunc float %x to bfloat
  %above = fcmp ogt bfloat %narrow, 0xR7F00
  ret i1 %above
}

define i1 @below_neg_bfloat(float %x) {
; CHECK-LABEL: @below_neg_bfloat(
; CHECK: fcmp olt float %x, f0xFF008000
  %narrow = fptrunc float %x to bfloat
  %below = fcmp olt bfloat %narrow, 0xRFF00
  ret i1 %below
}
