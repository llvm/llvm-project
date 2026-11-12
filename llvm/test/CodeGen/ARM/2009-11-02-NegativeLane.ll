; RUN: llc -mcpu=cortex-a8 -arm-atomic-cfg-tidy=0 < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-eabi"

define arm_aapcs_vfpcc void @foo(ptr nocapture %pBuffer, i32 %numItems) nounwind {
entry:
  br i1 undef, label %return, label %bb

bb:                                               ; preds = %bb, %entry
; CHECK: vld1.16 {d16[], d17[]}
  %0 = load i16, ptr undef, align 2
  %1 = insertelement <8 x i16> undef, i16 %0, i32 2
  %2 = insertelement <8 x i16> %1, i16 undef, i32 3
  %3 = mul <8 x i16> %2, %2
  %4 = extractelement <8 x i16> %3, i32 2
  store i16 %4, ptr undef, align 2
  br i1 undef, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}
