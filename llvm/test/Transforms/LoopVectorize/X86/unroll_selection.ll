; RUN: opt < %s  -loop-vectorize -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx -force-vector-width=4 -force-vector-interleave=0 -dce -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Don't unroll when we have register pressure.
;CHECK: reg_pressure
;CHECK: load <4 x double>
;CHECK-NOT: load  <4 x double>
;CHECK: store <4 x double>
;CHECK-NOT: store <4 x double>
;CHECK: ret
define void @reg_pressure(double* nocapture %A, i32 %n) nounwind uwtable ssp {
  %1 = sext i32 %n to i64
  br label %2

; <label>:2                                       ; preds = %2, %0
  %indvars.iv = phi i64 [ %indvars.iv.next, %2 ], [ %1, %0 ]
  %3 = getelementptr inbounds double, double* %A, i64 %indvars.iv
  %4 = load double, double* %3, align 8
  %5 = fadd double %4, 3.000000e+00
  %6 = fmul double %4, 2.000000e+00
  %7 = fadd double %5, %6
  %8 = fadd double %7, 2.000000e+00
  %9 = fmul double %8, 5.000000e-01
  %10 = fadd double %6, %9
  %11 = fsub double %10, %5
  %12 = fadd double %4, %11
  %13 = fdiv double %8, %12
  %14 = fmul double %13, %8
  %15 = fmul double %6, %14
  %16 = fmul double %5, %15
  %17 = fadd double %16, -3.000000e+00
  %18 = fsub double %4, %5
  %19 = fadd double %6, %18
  %20 = fadd double %13, %19
  %21 = fadd double %20, %17
  %22 = fadd double %21, 3.000000e+00
  %23 = fmul double %4, %22
  store double %23, double* %3, align 8
  %indvars.iv.next = add i64 %indvars.iv, -1
  %24 = trunc i64 %indvars.iv to i32
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %26, label %2

; <label>:26                                      ; preds = %2
  ret void
}

; This is a small loop. Unroll it twice. 
;CHECK: small_loop
;CHECK: xor
;CHECK: xor
;CHECK: ret
define void @small_loop(i16* nocapture %A, i64 %n) nounwind uwtable ssp {
  %1 = icmp eq i64 %n, 0
  br i1 %1, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %0, %.lr.ph
  %i.01 = phi i64 [ %5, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i16, i16* %A, i64 %i.01
  %3 = load i16, i16* %2, align 2
  %4 = xor i16 %3, 3
  store i16 %4, i16* %2, align 2
  %5 = add i64 %i.01, 1
  %exitcond = icmp eq i64 %5, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret void
}
