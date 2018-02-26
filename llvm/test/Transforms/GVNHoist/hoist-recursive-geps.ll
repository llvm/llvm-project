; RUN: opt -gvn-hoist -newgvn -gvn-hoist -S < %s | FileCheck %s

; Check that recursive GEPs are hoisted. Since hoisting creates
; fully redundant instructions, newgvn is run to remove them which then
; creates more opportunites for hoisting.

; CHECK-LABEL: @fun
; CHECK: load
; CHECK: fdiv
; CHECK: load
; CHECK: load
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK: fsub
; CHECK: fmul
; CHECK-NOT: fsub
; CHECK-NOT: fmul

%0 = type { double, double, double }
%1 = type { double, double, double }
%2 = type { %3, %1, %1 }
%3 = type { i32 (...)**, %4, %10*, %11, %11, %11, %11, %11, %11, %11, %11, %11 }
%4 = type { %5 }
%5 = type { %6 }
%6 = type { %7 }
%7 = type { %8 }
%8 = type { %9 }
%9 = type { i64, i64, i8* }
%10 = type <{ i32 (...)**, i32, [4 x i8] }>
%11 = type { [4 x [4 x double]] }
%12 = type <{ %1, %0, i32, [4 x i8] }>
%13 = type { %1, %0, %12, %3*, %14* }
%14 = type opaque

@d = external global %0, align 8
@p = external global %1, align 8

define zeroext i1 @fun(%2*, %12* dereferenceable(56), double*, %13*) {
  %5 = alloca %2*, align 8
  %6 = alloca %12*, align 8
  %7 = alloca double*, align 8
  %8 = alloca %13*, align 8
  %9 = alloca double, align 8
  %10 = alloca double, align 8
  %11 = alloca double, align 8
  %12 = alloca double, align 8
  %13 = alloca double, align 8
  %14 = alloca double, align 8
  %15 = alloca double, align 8
  store %2* %0, %2** %5, align 8
  store %12* %1, %12** %6, align 8
  store double* %2, double** %7, align 8
  store %13* %3, %13** %8, align 8
  %16 = load %2*, %2** %5, align 8
  %17 = load double, double* getelementptr inbounds (%0, %0* @d, i32 0, i32 0), align 8
  %18 = fdiv double 1.000000e+00, %17
  store double %18, double* %15, align 8
  %19 = load double, double* %15, align 8
  %20 = fcmp oge double %19, 0.000000e+00
  br i1 %20, label %21, label %36

; <label>:21:                                     ; preds = %4
  %22 = getelementptr inbounds %2, %2* %16, i32 0, i32 1
  %23 = getelementptr inbounds %1, %1* %22, i32 0, i32 0
  %24 = load double, double* %23, align 8
  %25 = load double, double* getelementptr inbounds (%1, %1* @p, i32 0, i32 0), align 8
  %26 = fsub double %24, %25
  %27 = load double, double* %15, align 8
  %28 = fmul double %26, %27
  store double %28, double* %9, align 8
  %29 = getelementptr inbounds %2, %2* %16, i32 0, i32 2
  %30 = getelementptr inbounds %1, %1* %29, i32 0, i32 0
  %31 = load double, double* %30, align 8
  %32 = load double, double* getelementptr inbounds (%1, %1* @p, i32 0, i32 0), align 8
  %33 = fsub double %31, %32
  %34 = load double, double* %15, align 8
  %35 = fmul double %33, %34
  store double %35, double* %12, align 8
  br label %51

; <label>:36:                                     ; preds = %4
  %37 = getelementptr inbounds %2, %2* %16, i32 0, i32 2
  %38 = getelementptr inbounds %1, %1* %37, i32 0, i32 0
  %39 = load double, double* %38, align 8
  %40 = load double, double* getelementptr inbounds (%1, %1* @p, i32 0, i32 0), align 8
  %41 = fsub double %39, %40
  %42 = load double, double* %15, align 8
  %43 = fmul double %41, %42
  store double %43, double* %9, align 8
  %44 = getelementptr inbounds %2, %2* %16, i32 0, i32 1
  %45 = getelementptr inbounds %1, %1* %44, i32 0, i32 0
  %46 = load double, double* %45, align 8
  %47 = load double, double* getelementptr inbounds (%1, %1* @p, i32 0, i32 0), align 8
  %48 = fsub double %46, %47
  %49 = load double, double* %15, align 8
  %50 = fmul double %48, %49
  store double %50, double* %12, align 8
  br label %51

; <label>:51:                                     ; preds = %36, %21
  %52 = load double, double* %12, align 8
  %53 = load double, double* %9, align 8
  %54 = fcmp olt double %52, %53
  ret i1 %54
}
