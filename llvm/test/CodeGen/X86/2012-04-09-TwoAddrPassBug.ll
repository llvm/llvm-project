; RUN: llc -O1 -verify-coalescing < %s
; PR12495
target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

define void @func(i8* nocapture) nounwind uwtable ssp align 2 {
  br i1 undef, label %4, label %2

; <label>:2                                       ; preds = %1                  
  %3 = tail call double @foo() nounwind
  br label %4

; <label>:4                                       ; preds = %2, %1              
  %5 = phi double [ %3, %2 ], [ 0.000000e+00, %1 ]
  %6 = fsub double %5, undef
  %7 = fcmp olt double %6, 0.000000e+00
  %8 = select i1 %7, double 0.000000e+00, double %6
  %9 = fcmp olt double undef, 0.000000e+00
  %10 = fcmp olt double %8, undef
  %11 = or i1 %9, %10
  br i1 %11, label %12, label %14

; <label>:12                                      ; preds = %4                  
  %13 = tail call double @fmod(double %8, double 0.000000e+00) nounwind
  unreachable

; <label>:14                                      ; preds = %4                  
  ret void
}

declare double @foo()

declare double @fmod(double, double)
