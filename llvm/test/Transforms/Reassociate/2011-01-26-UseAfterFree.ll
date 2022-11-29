; RUN: opt < %s -passes=reassociate
; PR9039
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-gnu-linux"

define void @exp_averages_intraday__deviation() {
entry:
  %0 = load i32, ptr undef, align 4
  %1 = shl i32 %0, 2
  %2 = add nsw i32 undef, %1
  %3 = add nsw i32 %2, undef
  %4 = mul nsw i32 %0, 12
  %5 = add nsw i32 %3, %4
  %6 = add nsw i32 %5, %4
  %7 = add nsw i32 %6, undef
  br i1 false, label %"4", label %"12"

"4":                                              ; preds = %entry
  br i1 undef, label %"5", label %"8"

"5":                                              ; preds = %"4"
  unreachable

"8":                                              ; preds = %"4"
  %8 = getelementptr inbounds i8, ptr undef, i32 %6
  br i1 undef, label %"13", label %"12"

"12":                                             ; preds = %"8", %entry
  ret void

"13":                                             ; preds = %"8"
  ret void
}
