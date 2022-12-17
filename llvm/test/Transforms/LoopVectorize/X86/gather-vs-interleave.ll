; RUN: opt -passes=loop-vectorize -S -mattr=avx512f  < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This test checks that "gather" operation is choosen since it's cost is better
; than interleaving pattern.
;
;unsigned long A[SIZE];
;unsigned long B[SIZE];
;
;void foo() {
;  for (int i=0; i<N; i+=8) {
;    B[i] = A[i] + 5;
;  }
;}

@A = global [10240 x i64] zeroinitializer, align 16
@B = global [10240 x i64] zeroinitializer, align 16


; CHECK-LABEL: strided_load_i64
; CHECK: masked.gather
define void @strided_load_i64() {
  br label %1

; <label>:1:                                      ; preds = %0, %1
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds [10240 x i64], [10240 x i64]* @A, i64 0, i64 %indvars.iv
  %3 = load i64, i64* %2, align 16
  %4 = add i64 %3, 5
  %5 = getelementptr inbounds [10240 x i64], [10240 x i64]* @B, i64 0, i64 %indvars.iv
  store i64 %4, i64* %5, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 8
  %6 = icmp slt i64 %indvars.iv.next, 1024
  br i1 %6, label %1, label %7

; <label>:7:                                      ; preds = %1
  ret void
}

