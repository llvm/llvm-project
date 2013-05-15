; RUN: opt %loadPolly %defaultOpts -polly-cloog -analyze < %s | FileCheck %s

;#include <string.h>
;#define N 1024
;int A[N];
;int B[N];
;
;void loop_with_condition_ineq() {
;  int i;
;
;  __sync_synchronize();
;  for (i = 0; i < N; i++) {
;    if (i != N / 2)
;      A[i] = 1;
;    else
;      A[i] = 2;
;    B[i] = 3;
;  }
;  __sync_synchronize();
;}
;
;int main () {
;  int i;
;
;  memset(A, 0, sizeof(int) * N);
;  memset(B, 0, sizeof(int) * N);
;
;  loop_with_condition_ineq();
;
;  for (i = 0; i < N; i++)
;    if (B[i] != 3)
;      return 1;
;
;  for (i = 0; i < N; i++)
;    if (i != N / 2 && A[i] != 1)
;      return 1;
;    else if (i == N && A[i] != 2)
;      return 1;
;  return 0;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

@A = common global [1024 x i32] zeroinitializer, align 16 ; <[1024 x i32]*> [#uses=4]
@B = common global [1024 x i32] zeroinitializer, align 16 ; <[1024 x i32]*> [#uses=4]

define void @loop_with_condition_ineq() nounwind {
; <label>:0
  fence seq_cst
  br label %1

; <label>:1                                       ; preds = %7, %0
  %indvar = phi i64 [ %indvar.next, %7 ], [ 0, %0 ] ; <i64> [#uses=5]
  %scevgep = getelementptr [1024 x i32]* @A, i64 0, i64 %indvar ; <i32*> [#uses=2]
  %scevgep1 = getelementptr [1024 x i32]* @B, i64 0, i64 %indvar ; <i32*> [#uses=1]
  %i.0 = trunc i64 %indvar to i32                 ; <i32> [#uses=1]
  %exitcond = icmp ne i64 %indvar, 1024           ; <i1> [#uses=1]
  br i1 %exitcond, label %2, label %8

; <label>:2                                       ; preds = %1
  %3 = icmp ne i32 %i.0, 512                      ; <i1> [#uses=1]
  br i1 %3, label %4, label %5

; <label>:4                                       ; preds = %2
  store i32 1, i32* %scevgep
  br label %6

; <label>:5                                       ; preds = %2
  store i32 2, i32* %scevgep
  br label %6

; <label>:6                                       ; preds = %5, %4
  store i32 3, i32* %scevgep1
  br label %7

; <label>:7                                       ; preds = %6
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %1

; <label>:8                                       ; preds = %1
  fence seq_cst
  ret void
}

define i32 @main() nounwind {
; <label>:0
  call void @llvm.memset.p0i8.i64(i8* bitcast ([1024 x i32]* @A to i8*), i8 0, i64 4096, i32 1, i1 false)
  call void @llvm.memset.p0i8.i64(i8* bitcast ([1024 x i32]* @B to i8*), i8 0, i64 4096, i32 1, i1 false)
  call void @loop_with_condition_ineq()
  br label %1

; <label>:1                                       ; preds = %8, %0
  %indvar1 = phi i64 [ %indvar.next2, %8 ], [ 0, %0 ] ; <i64> [#uses=3]
  %scevgep3 = getelementptr [1024 x i32]* @B, i64 0, i64 %indvar1 ; <i32*> [#uses=1]
  %i.0 = trunc i64 %indvar1 to i32                ; <i32> [#uses=1]
  %2 = icmp slt i32 %i.0, 1024                    ; <i1> [#uses=1]
  br i1 %2, label %3, label %9

; <label>:3                                       ; preds = %1
  %4 = load i32* %scevgep3                        ; <i32> [#uses=1]
  %5 = icmp ne i32 %4, 3                          ; <i1> [#uses=1]
  br i1 %5, label %6, label %7

; <label>:6                                       ; preds = %3
  br label %28

; <label>:7                                       ; preds = %3
  br label %8

; <label>:8                                       ; preds = %7
  %indvar.next2 = add i64 %indvar1, 1             ; <i64> [#uses=1]
  br label %1

; <label>:9                                       ; preds = %1
  br label %10

; <label>:10                                      ; preds = %26, %9
  %indvar = phi i64 [ %indvar.next, %26 ], [ 0, %9 ] ; <i64> [#uses=3]
  %scevgep = getelementptr [1024 x i32]* @A, i64 0, i64 %indvar ; <i32*> [#uses=2]
  %i.1 = trunc i64 %indvar to i32                 ; <i32> [#uses=3]
  %11 = icmp slt i32 %i.1, 1024                   ; <i1> [#uses=1]
  br i1 %11, label %12, label %27

; <label>:12                                      ; preds = %10
  %13 = icmp ne i32 %i.1, 512                     ; <i1> [#uses=1]
  br i1 %13, label %14, label %18

; <label>:14                                      ; preds = %12
  %15 = load i32* %scevgep                        ; <i32> [#uses=1]
  %16 = icmp ne i32 %15, 1                        ; <i1> [#uses=1]
  br i1 %16, label %17, label %18

; <label>:17                                      ; preds = %14
  br label %28

; <label>:18                                      ; preds = %14, %12
  %19 = icmp eq i32 %i.1, 1024                    ; <i1> [#uses=1]
  br i1 %19, label %20, label %24

; <label>:20                                      ; preds = %18
  %21 = load i32* %scevgep                        ; <i32> [#uses=1]
  %22 = icmp ne i32 %21, 2                        ; <i1> [#uses=1]
  br i1 %22, label %23, label %24

; <label>:23                                      ; preds = %20
  br label %28

; <label>:24                                      ; preds = %20, %18
  br label %25

; <label>:25                                      ; preds = %24
  br label %26

; <label>:26                                      ; preds = %25
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %10

; <label>:27                                      ; preds = %10
  br label %28

; <label>:28                                      ; preds = %27, %23, %17, %6
  %.0 = phi i32 [ 1, %6 ], [ 1, %17 ], [ 1, %23 ], [ 0, %27 ] ; <i32> [#uses=1]
  ret i32 %.0
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind

; CHECK: for (c2=0;c2<=511;c2++) {
; CHECK:     Stmt_4(c2);
; CHECK:       Stmt_6(c2);
; CHECK: }
; CHECK: Stmt_5(512);
; CHECK: Stmt_6(512);
; CHECK: for (c2=513;c2<=1023;c2++) {
; CHECK:     Stmt_4(c2);
; CHECK:       Stmt_6(c2);
; CHECK: }

