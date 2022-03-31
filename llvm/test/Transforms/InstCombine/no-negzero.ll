; RUN: opt < %s -passes=instcombine -S | FileCheck %s
; ModuleID = '3555a.c'
; sqrt(fabs) cannot be negative zero, so we should eliminate the fadd.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.8"

; CHECK-LABEL: @mysqrt(
; CHECK-NOT: fadd
; CHECK: ret
define double @mysqrt(double %x) nounwind {
entry:
  %x_addr = alloca double                         ; <double*> [#uses=2]
  %retval = alloca double, align 8                ; <double*> [#uses=2]
  %0 = alloca double, align 8                     ; <double*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store double %x, double* %x_addr
  %1 = load double, double* %x_addr, align 8              ; <double> [#uses=1]
  %2 = call double @fabs(double %1) nounwind readnone ; <double> [#uses=1]
  %3 = call double @sqrt(double %2) nounwind readonly ; <double> [#uses=1]
  %4 = fadd double %3, 0.000000e+00               ; <double> [#uses=1]
  store double %4, double* %0, align 8
  %5 = load double, double* %0, align 8                   ; <double> [#uses=1]
  store double %5, double* %retval, align 8
  br label %return

return:                                           ; preds = %entry
  %retval1 = load double, double* %retval                 ; <double> [#uses=1]
  ret double %retval1
}

declare double @fabs(double)

declare double @sqrt(double) nounwind readonly
