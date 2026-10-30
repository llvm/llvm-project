; fptrunc float -> bfloat is integer rounding in the Metal pipeline. A NaN
; keeps its high bits with the quiet bit set: the rounding add would carry out
; of 0x7fffffff into the sign (-0; PTX-formalization cvt-10 cvt.rn.relu.bf16.f32
; gave 0 for NaN on an M2 Max). NaN is found with integer operations only.
; RUN: opt -mtriple=air -passes=air-bfloat16-cast-decompose -S %s | FileCheck %s

; CHECK-LABEL: define bfloat @t(
; CHECK: [[B:%.*]] = bitcast float %x to i32
; CHECK: [[R:%bf16_rnd]] = add i32 [[B]],
; CHECK: [[A:%.*]] = and i32 [[B]], 2147483647
; CHECK: [[N:%.*]] = icmp ugt i32 [[A]], 2139095040
; CHECK: [[H:%.*]] = lshr i32 [[B]], 16
; CHECK: [[Q:%.*]] = or i32 [[H]], 64
; CHECK: [[RH:%.*]] = lshr i32 [[R]], 16
; CHECK: select i1 [[N]], i32 [[Q]], i32 [[RH]]
; CHECK-NOT: fcmp
define bfloat @t(float %x) {
  %r = fptrunc float %x to bfloat
  ret bfloat %r
}
