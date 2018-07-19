; RUN: opt < %s -basicaa -slp-vectorizer -S | FileCheck %s
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "x86_64--linux-gnu"

@a = common global [4 x double] zeroinitializer, align 8
@b = common global [4 x double] zeroinitializer, align 8

; [4], b[4];
; double foo() {
;  double sum =0;
;  sum = (a[0]+b[0]) + (a[1]+b[1]);
;  return sum;
; }

; CHECK-LABEL: @return1
; CHECK: %0 = load <2 x double>, <2 x double>*
; CHECK: %1 = load <2 x double>, <2 x double>*
; CHECK: %2 = fadd <2 x double>

define double @return1() {
entry:
  %a0 = load double, double* getelementptr inbounds ([4 x double], [4 x double]* @a, i32 0, i32 0), align 8
  %b0 = load double, double* getelementptr inbounds ([4 x double], [4 x double]* @b, i32 0, i32 0), align 8
  %add0 = fadd double %a0, %b0
  %a1 = load double, double* getelementptr inbounds ([4 x double], [4 x double]* @a, i32 0, i32 1), align 8
  %b1 = load double, double* getelementptr inbounds ([4 x double], [4 x double]* @b, i32 0, i32 1), align 8
  %add1 = fadd double %a1, %b1
  %add2 = fadd double %add0, %add1
  ret double %add2
}

; double hadd(double *x) {
;   return ((x[0] + x[2]) + (x[1] + x[3]));
; }

; CHECK-LABEL: @return2
; CHECK: %1 = load <2 x double>, <2 x double>*
; CHECK: %3 = load <2 x double>, <2 x double>* %2
; CHECK: %4 = fadd <2 x double> %1, %3

define double @return2(double* nocapture readonly %x) {
entry:
  %x0 = load double, double* %x, align 4
  %arrayidx1 = getelementptr inbounds double, double* %x, i32 2
  %x2 = load double, double* %arrayidx1, align 4
  %add3 = fadd double %x0, %x2
  %arrayidx2 = getelementptr inbounds double, double* %x, i32 1
  %x1 = load double, double* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds double, double* %x, i32 3
  %x3 = load double, double* %arrayidx3, align 4
  %add4 = fadd double %x1, %x3
  %add5 = fadd double %add3, %add4
  ret double %add5
}
