; RUN: llc < %s -verify-coalescing
; PR11861
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv7-none-linux-eabi"

define arm_aapcs_vfpcc void @foo() nounwind uwtable align 2 {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %2 = phi <4 x float> [ undef, %0 ], [ %11, %1 ]
  %3 = bitcast <4 x float> %2 to <2 x i64>
  %4 = shufflevector <2 x i64> %3, <2 x i64> undef, <1 x i32> zeroinitializer
  %5 = xor <2 x i32> zeroinitializer, <i32 -1, i32 -1>
  %6 = bitcast <2 x i32> zeroinitializer to <2 x float>
  %7 = shufflevector <2 x float> zeroinitializer, <2 x float> %6, <2 x i32> <i32 0, i32 2>
  %8 = shufflevector <2 x i64> %3, <2 x i64> undef, <1 x i32> <i32 1>
  %9 = bitcast <2 x float> %7 to <1 x i64>
  %10 = shufflevector <1 x i64> %9, <1 x i64> %8, <2 x i32> <i32 0, i32 1>
  %11 = bitcast <2 x i64> %10 to <4 x float>
  br label %1
}
