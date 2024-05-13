; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -S -enable-if-conversion | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; This is kernel11 from "LivermoreLoops". We can't vectorize it because we
; access both x[k] and x[k-1].
;
; void kernel11(ptr x, ptr y, int n) {
;   for ( int k=1 ; k<n ; k++ )
;     x[k] = x[k-1] + y[k];
; }

; CHECK-LABEL: @kernel11(
; CHECK-NOT: <4 x double>
; CHECK: ret
define i32 @kernel11(ptr %x, ptr %y, i32 %n) nounwind uwtable ssp {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %k = alloca i32, align 4
  store ptr %x, ptr %1, align 8
  store ptr %y, ptr %2, align 8
  store i32 %n, ptr %3, align 4
  store i32 1, ptr %k, align 4
  br label %4

; <label>:4                                       ; preds = %25, %0
  %5 = load i32, ptr %k, align 4
  %6 = load i32, ptr %3, align 4
  %7 = icmp slt i32 %5, %6
  br i1 %7, label %8, label %28

; <label>:8                                       ; preds = %4
  %9 = load i32, ptr %k, align 4
  %10 = sub nsw i32 %9, 1
  %11 = sext i32 %10 to i64
  %12 = load ptr, ptr %1, align 8
  %13 = getelementptr inbounds double, ptr %12, i64 %11
  %14 = load double, ptr %13, align 8
  %15 = load i32, ptr %k, align 4
  %16 = sext i32 %15 to i64
  %17 = load ptr, ptr %2, align 8
  %18 = getelementptr inbounds double, ptr %17, i64 %16
  %19 = load double, ptr %18, align 8
  %20 = fadd double %14, %19
  %21 = load i32, ptr %k, align 4
  %22 = sext i32 %21 to i64
  %23 = load ptr, ptr %1, align 8
  %24 = getelementptr inbounds double, ptr %23, i64 %22
  store double %20, ptr %24, align 8
  br label %25

; <label>:25                                      ; preds = %8
  %26 = load i32, ptr %k, align 4
  %27 = add nsw i32 %26, 1
  store i32 %27, ptr %k, align 4
  br label %4

; <label>:28                                      ; preds = %4
  ret i32 0
}


; A[i*7] is scalarized, and the different scalars can in theory wrap
; around and overwrite other scalar elements. However we can still
; vectorize because we can version the loop to avoid this case.
;
; void foo(int *a) {
;   for (int i=0; i<256; ++i) {
;     int x = a[i*7];
;     if (x>3)
;       x = x*x+x*4;
;     a[i*7] = x+3;
;   }
; }

; CHECK-LABEL: @func2(
; CHECK: <4 x i32>
; CHECK: ret
define i32 @func2(ptr nocapture %a) nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %7, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %7 ]
  %2 = mul nsw i64 %indvars.iv, 7
  %3 = getelementptr inbounds i32, ptr %a, i64 %2
  %4 = load i32, ptr %3, align 4
  %5 = icmp sgt i32 %4, 3
  br i1 %5, label %6, label %7

; <label>:6                                       ; preds = %1
  %tmp = add i32 %4, 4
  %tmp1 = mul i32 %tmp, %4
  br label %7

; <label>:7                                       ; preds = %6, %1
  %x.0 = phi i32 [ %tmp1, %6 ], [ %4, %1 ]
  %8 = add nsw i32 %x.0, 3
  store i32 %8, ptr %3, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 256
  br i1 %exitcond, label %9, label %1

; <label>:9                                       ; preds = %7
  ret i32 0
}
