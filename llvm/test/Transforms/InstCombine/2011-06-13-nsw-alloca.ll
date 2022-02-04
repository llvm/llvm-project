; RUN: opt -S -passes=instcombine < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0.0"

; CHECK-LABEL: define void @fu1(
define void @fu1(i32 %parm) nounwind ssp {
  %1 = alloca i32, align 4
; CHECK: alloca double*
  %ptr = alloca double*, align 4
  store i32 %parm, i32* %1, align 4
  store double* null, double** %ptr, align 4
  %2 = load i32, i32* %1, align 4
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %4, label %10

; <label>:4                                       ; preds = %0
  %5 = load i32, i32* %1, align 4
  %6 = shl nsw i32 %5, 3
; With "nsw", the alloca and its bitcast can be fused:
  %7 = add nsw i32 %6, 2048
;  CHECK: alloca double
  %8 = alloca i8, i32 %7
  %9 = bitcast i8* %8 to double*
; CHECK-NEXT: store double*
  store double* %9, double** %ptr, align 4
  br label %10
; <label>:10                                      ; preds = %4, %0
  %11 = load double*, double** %ptr, align 4
  call void @bar(double* %11)
; CHECK: ret
  ret void
}

declare void @bar(double*)

; CHECK-LABEL: define void @fu2(
define void @fu2(i32 %parm) nounwind ssp {
  %1 = alloca i32, align 4
  %ptr = alloca double*, align 4
  store i32 %parm, i32* %1, align 4
  store double* null, double** %ptr, align 4
  %2 = load i32, i32* %1, align 4
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %4, label %10

; <label>:4                                       ; preds = %0
  %5 = load i32, i32* %1, align 4
  %6 = mul nsw i32 %5, 8
; Without "nsw", the alloca and its bitcast cannot be fused:
  %7 = add  i32 %6, 2048
; CHECK: alloca i8
  %8 = alloca i8, i32 %7
; CHECK-NEXT: bitcast double**
; CHECK-NEXT: store i8*
  %9 = bitcast i8* %8 to double*
  store double* %9, double** %ptr, align 4
  br label %10

; <label>:10                                      ; preds = %4, %0
  %11 = load double*, double** %ptr, align 4
  call void @bar(double* %11)
  ret void
}

