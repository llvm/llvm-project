; RUN: opt -passes=instcombine -S %s | FileCheck %s

define i1 @peek_through_bitcast_signbit(float %f) {
; CHECK-LABEL: @peek_through_bitcast_signbit
; CHECK: %cmp = fcmp olt float %f, 0.000000e+00
; CHECK: ret i1 %cmp

  %bc = bitcast float %f to i32
  %cmp = icmp slt i32 %bc, 0
  ret i1 %cmp
}
