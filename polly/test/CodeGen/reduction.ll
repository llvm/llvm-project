; RUN: opt %loadPolly -polly-codegen -S < %s 2>&1 | not FileCheck %s

;#include <string.h>
;#include <stdio.h>
;#define N 1021
;
;int main () {
;  int i;
;  int A[N];
;  int red;
;
;  memset(A, 0, sizeof(int) * N);
;
;  A[0] = 1;
;  A[1] = 1;
;  red = 0;
;
;  __sync_synchronize();
;
;  for (i = 2; i < N; i++) {
;    A[i] = A[i-1] + A[i-2];
;    red += A[i-2];
;  }
;
;  __sync_synchronize();
;
;  if (red != 382399368)
;    return 1;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define i32 @main() nounwind {
; <label>:0
  %A = alloca [1021 x i32], align 16              ; <ptr> [#uses=6]
  %1 = getelementptr inbounds [1021 x i32], ptr %A, i32 0, i32 0 ; <ptr> [#uses=1]
  call void @llvm.memset.p0.i64(ptr %1, i8 0, i64 4084, i32 1, i1 false)
  %2 = getelementptr inbounds [1021 x i32], ptr %A, i32 0, i32 0 ; <ptr> [#uses=1]
  %3 = getelementptr inbounds i32, ptr %2, i64 0      ; <ptr> [#uses=1]
  store i32 1, ptr %3
  %4 = getelementptr inbounds [1021 x i32], ptr %A, i32 0, i32 0 ; <ptr> [#uses=1]
  %5 = getelementptr inbounds i32, ptr %4, i64 1      ; <ptr> [#uses=1]
  store i32 1, ptr %5
  fence seq_cst
  br label %6

; <label>:7                                       ; preds = %13, %0
  %indvar = phi i64 [ %indvar.next, %13 ], [ 0, %0 ] ; <i64> [#uses=5]
  %red.0 = phi i32 [ 0, %0 ], [ %12, %13 ]        ; <i32> [#uses=2]
  %scevgep = getelementptr [1021 x i32], ptr %A, i64 0, i64 %indvar ; <ptr> [#uses=2]
  %tmp = add i64 %indvar, 2                       ; <i64> [#uses=1]
  %scevgep1 = getelementptr [1021 x i32], ptr %A, i64 0, i64 %tmp ; <ptr> [#uses=1]
  %tmp2 = add i64 %indvar, 1                      ; <i64> [#uses=1]
  %scevgep3 = getelementptr [1021 x i32], ptr %A, i64 0, i64 %tmp2 ; <ptr> [#uses=1]
  %exitcond = icmp ne i64 %indvar, 1019           ; <i1> [#uses=1]
  br i1 %exitcond, label %7, label %14

; <label>:8                                       ; preds = %6
  %8 = load i32, ptr %scevgep3                        ; <i32> [#uses=1]
  %9 = load i32, ptr %scevgep                        ; <i32> [#uses=1]
  %10 = add nsw i32 %8, %9                       ; <i32> [#uses=1]
  store i32 %10, ptr %scevgep1
  %11 = load i32, ptr %scevgep                        ; <i32> [#uses=1]
  %12 = add nsw i32 %red.0, %11                   ; <i32> [#uses=1]
  br label %13

; <label>:14                                      ; preds = %7
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %6

; <label>:15                                      ; preds = %6
  %red.0.lcssa = phi i32 [ %red.0, %6 ]           ; <i32> [#uses=1]
  fence seq_cst
  %15 = icmp ne i32 %red.0.lcssa, 382399368       ; <i1> [#uses=1]
  br i1 %15, label %16, label %17

; <label>:17                                      ; preds = %14
  br label %17

; <label>:18                                      ; preds = %16, %14
  %.0 = phi i32 [ 1, %16 ], [ 0, %14 ]            ; <i32> [#uses=1]
  ret i32 %.0
}

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i32, i1) nounwind

; CHECK:  Could not generate independent blocks
