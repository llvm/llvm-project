; RUN: llc < %s | FileCheck %s
; RUN: llc -mattr=+vfp3,+fp16 < %s | FileCheck --check-prefix=CHECK-FP16 %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "armv7-eabi"

@x = global i16 12902
@y = global i16 0
@z = common global i16 0

define arm_aapcs_vfpcc void @foo() nounwind {
; CHECK: foo:
; CHECK-FP6: foo:
entry:
  %0 = load i16* @x, align 2
  %1 = load i16* @y, align 2
  %2 = tail call float @llvm.convert.from.fp16(i16 %0)
; CHECK: __gnu_h2f_ieee
; CHECK-FP16: vcvtb.f32.f16
  %3 = tail call float @llvm.convert.from.fp16(i16 %1)
; CHECK: __gnu_h2f_ieee
; CHECK-FP16: vcvtb.f32.f16
  %4 = fadd float %2, %3
  %5 = tail call i16 @llvm.convert.to.fp16(float %4)
; CHECK: __gnu_f2h_ieee
; CHECK-FP16: vcvtb.f16.f32
  store i16 %5, i16* @x, align 2
  ret void
}

declare float @llvm.convert.from.fp16(i16) nounwind readnone

declare i16 @llvm.convert.to.fp16(float) nounwind readnone
