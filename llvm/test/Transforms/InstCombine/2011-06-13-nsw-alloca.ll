; RUN: opt -S -passes=instcombine < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0.0"

; CHECK-LABEL: define void @fu1(
define void @fu1(i32 %parm) #0 {
bb:
  %i = alloca i32, align 4

  ; CHECK: alloca double*
  %ptr = alloca double*, align 4
  store i32 %parm, i32* %i, align 4
  store double* null, double** %ptr, align 4
  %i1 = load i32, i32* %i, align 4
  %i2 = icmp ne i32 %i1, 0
  br i1 %i2, label %bb3, label %bb9

bb3:                                              ; preds = %bb
  %i4 = load i32, i32* %i, align 4
  %i5 = shl nuw i32 %i4, 3
  ; With "nuw", the alloca and its bitcast can be fused:
  %i6 = add nuw i32 %i5, 2048

  ;  CHECK: alloca double
  %i7 = alloca i8, i32 %i6, align 1
  %i8 = bitcast i8* %i7 to double*
  ; CHECK-NEXT: store double*
  store double* %i8, double** %ptr, align 4
  br label %bb9

bb9:                                              ; preds = %bb3, %bb
  %i10 = load double*, double** %ptr, align 4
  call void @bar(double* %i10)
  ret void
}

declare void @bar(double*)

; CHECK-LABEL: define void @fu2(
define void @fu2(i32 %parm) #0 {
bb:
  %i = alloca i32, align 4
  %ptr = alloca double*, align 4
  store i32 %parm, i32* %i, align 4
  store double* null, double** %ptr, align 4
  %i1 = load i32, i32* %i, align 4
  %i2 = icmp ne i32 %i1, 0
  br i1 %i2, label %bb3, label %bb9

bb3:                                              ; preds = %bb
  %i4 = load i32, i32* %i, align 4
  %i5 = mul nuw i32 %i4, 8
  ; Without "nuw", the alloca and its bitcast cannot be fused:
  %i6 = add i32 %i5, 2048
  ; CHECK: alloca i8
  %i7 = alloca i8, i32 %i6, align 1

  ; CHECK-NEXT: bitcast double**
  ; CHECK-NEXT: store i8*
  %i8 = bitcast i8* %i7 to double*
  store double* %i8, double** %ptr, align 4
  br label %bb9

bb9:                                              ; preds = %bb3, %bb
  %i10 = load double*, double** %ptr, align 4
  call void @bar(double* %i10)
  ret void
}

attributes #0 = { nounwind ssp }
