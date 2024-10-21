; RUN: opt -S --passes=slp-vectorizer -mtriple=s390x-unknown-linux -mcpu=z16 < %s | FileCheck %s
;
; Test functions that (at least currently) only gets vectorized if the
; insertion cost for an element load is counted as free.

; This function needs the free element load to be recognized in SLP
; getGatherCost().
define void @fun0(ptr nocapture %0, double %1) {
; CHECK-LABEL: define void @fun0(
; CHECK:    fmul <2 x double>
; CHECK:    call <2 x double> @llvm.fmuladd.v2f64(

  %3 = fmul double %1, 2.000000e+00
  %4 = tail call double @llvm.fmuladd.f64(double %3, double %3, double 0.000000e+00)
  %5 = tail call double @llvm.fmuladd.f64(double %3, double %3, double %4)
  %sqrt1 = tail call double @llvm.sqrt.f64(double %5)
  %6 = load double, ptr %0, align 8
  %7 = fmul double %6, 2.000000e+00
  %8 = tail call double @llvm.fmuladd.f64(double %7, double %7, double 0.000000e+00)
  %9 = tail call double @llvm.fmuladd.f64(double %7, double %7, double %8)
  %sqrt = tail call double @llvm.sqrt.f64(double %9)
  %10 = fadd double %sqrt1, %sqrt
  store double %10, ptr %0, align 8
  ret void
}


; This function needs the element-load to be recognized in SystemZ
; getVectorInstrCost().
define void @fun1(double %0) local_unnamed_addr {
; CHECK-LABEL: define void @fun1(
; CHECK:    fsub <2 x double>
; CHECK:    fsub <2 x double>
; CHECK:    fsub <2 x double>
; CHECK:    fmul <2 x double>
; CHECK:    call <2 x double> @llvm.fmuladd.v2f64(
; CHECK:    call <2 x double> @llvm.fmuladd.v2f64(
; CHECK:    %14 = fcmp olt <2 x double> %13, %2

  br label %2

2:
  %3 = phi double [ poison, %1 ], [ poison, %2 ]
  %4 = phi double [ undef, %1 ], [ poison, %2 ]
  %5 = phi double [ 0.000000e+00, %1 ], [ poison, %2 ]
  %6 = phi double [ 0.000000e+00, %1 ], [ poison, %2 ]
  %7 = phi double [ 0.000000e+00, %1 ], [ poison, %2 ]
  %8 = phi double [ 0.000000e+00, %1 ], [ %21, %2 ]
  %9 = fsub double 0.000000e+00, %8
  %10 = fsub double 0.000000e+00, %7
  %11 = fmul double %9, 0.000000e+00
  %12 = fmul double %10, 0.000000e+00
  %13 = fsub double 0.000000e+00, %6
  %14 = fsub double 0.000000e+00, %5
  %15 = tail call double @llvm.fmuladd.f64(double %13, double %13, double %11)
  %16 = tail call double @llvm.fmuladd.f64(double %14, double %14, double %12)
  %17 = fsub double 0.000000e+00, %4
  %18 = fsub double 0.000000e+00, %3
  %19 = tail call double @llvm.fmuladd.f64(double %17, double %17, double %15)
  %20 = tail call double @llvm.fmuladd.f64(double %18, double %18, double %16)
  %21 = load double, ptr null, align 8
  %22 = fcmp olt double %19, %0
  %23 = fcmp olt double %20, 0.000000e+00
  %24 = or i1 %23, %22
  br label %2
}

declare double @llvm.fmuladd.f64(double, double, double)

; This should *not* be vectorized as the insertion into the vector isn't free,
; which is recognized in SystemZTTImpl::getScalarizationOverhead().
define void @fun2(ptr %0, ptr %Dst) {
; CHECK-LABEL: define void @fun2(
; CHECK-NOT: store <2 x i64>
  %3 = load i64, ptr %0, align 8
  %4 = icmp eq i64 %3, 0
  br i1 %4, label %5, label %6

5:
  ret void

6:
  %7 = getelementptr i8, ptr %Dst, i64 24
  store i64 %3, ptr %7, align 8
  %8 = getelementptr i8, ptr %Dst, i64 16
  store i64 0, ptr %8, align 8
  br label %5
}
