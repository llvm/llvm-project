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
  %A = alloca [1021 x i32], align 16              ; <[1021 x i32]*> [#uses=6]
  %1 = getelementptr inbounds [1021 x i32], [1021 x i32]* %A, i32 0, i32 0 ; <i32*> [#uses=1]
  %2 = bitcast i32* %1 to i8*                     ; <i8*> [#uses=1]
  call void @llvm.memset.p0i8.i64(i8* %2, i8 0, i64 4084, i32 1, i1 false)
  %3 = getelementptr inbounds [1021 x i32], [1021 x i32]* %A, i32 0, i32 0 ; <i32*> [#uses=1]
  %4 = getelementptr inbounds i32, i32* %3, i64 0      ; <i32*> [#uses=1]
  store i32 1, i32* %4
  %5 = getelementptr inbounds [1021 x i32], [1021 x i32]* %A, i32 0, i32 0 ; <i32*> [#uses=1]
  %6 = getelementptr inbounds i32, i32* %5, i64 1      ; <i32*> [#uses=1]
  store i32 1, i32* %6
  fence seq_cst
  br label %7

; <label>:7                                       ; preds = %14, %0
  %indvar = phi i64 [ %indvar.next, %14 ], [ 0, %0 ] ; <i64> [#uses=5]
  %red.0 = phi i32 [ 0, %0 ], [ %13, %14 ]        ; <i32> [#uses=2]
  %scevgep = getelementptr [1021 x i32], [1021 x i32]* %A, i64 0, i64 %indvar ; <i32*> [#uses=2]
  %tmp = add i64 %indvar, 2                       ; <i64> [#uses=1]
  %scevgep1 = getelementptr [1021 x i32], [1021 x i32]* %A, i64 0, i64 %tmp ; <i32*> [#uses=1]
  %tmp2 = add i64 %indvar, 1                      ; <i64> [#uses=1]
  %scevgep3 = getelementptr [1021 x i32], [1021 x i32]* %A, i64 0, i64 %tmp2 ; <i32*> [#uses=1]
  %exitcond = icmp ne i64 %indvar, 1019           ; <i1> [#uses=1]
  br i1 %exitcond, label %8, label %15

; <label>:8                                       ; preds = %7
  %9 = load i32, i32* %scevgep3                        ; <i32> [#uses=1]
  %10 = load i32, i32* %scevgep                        ; <i32> [#uses=1]
  %11 = add nsw i32 %9, %10                       ; <i32> [#uses=1]
  store i32 %11, i32* %scevgep1
  %12 = load i32, i32* %scevgep                        ; <i32> [#uses=1]
  %13 = add nsw i32 %red.0, %12                   ; <i32> [#uses=1]
  br label %14

; <label>:14                                      ; preds = %8
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %7

; <label>:15                                      ; preds = %7
  %red.0.lcssa = phi i32 [ %red.0, %7 ]           ; <i32> [#uses=1]
  fence seq_cst
  %16 = icmp ne i32 %red.0.lcssa, 382399368       ; <i1> [#uses=1]
  br i1 %16, label %17, label %18

; <label>:17                                      ; preds = %15
  br label %18

; <label>:18                                      ; preds = %17, %15
  %.0 = phi i32 [ 1, %17 ], [ 0, %15 ]            ; <i32> [#uses=1]
  ret i32 %.0
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind

; CHECK:  Could not generate independent blocks
