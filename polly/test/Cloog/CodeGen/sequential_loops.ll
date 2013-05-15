; RUN: opt %loadPolly %defaultOpts -polly-cloog -analyze < %s | FileCheck %s

;#include <string.h>
;#define N 1024
;
;int A[N];
;
;void sequential_loops() {
;  int i;
;  for (i = 0; i < N/2; i++) {
;    A[i] = 1;
;  }
;  for (i = N/2 ; i < N; i++) {
;    A[i] = 2;
;  }
;}
;
;int main () {
;  int i;
;  memset(A, 0, sizeof(int) * N);
;
;  sequential_loops();
;
;  for (i = 0; i < N; i++) {
;    if (A[i] != 1 && i < N/2)
;      return 1;
;    if (A[i] !=  2 && i >= N/2)
;      return 1;
;  }
;
;  return 0;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

@A = common global [1024 x i32] zeroinitializer, align 4 ; <[1024 x i32]*> [#uses=5]

define void @sequential_loops() nounwind {
bb:
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  %indvar1 = phi i64 [ %indvar.next2, %bb3 ], [ 0, %bb ]
  %scevgep4 = getelementptr [1024 x i32]* @A, i64 0, i64 %indvar1
  %exitcond3 = icmp ne i64 %indvar1, 512
  br i1 %exitcond3, label %bb2, label %bb4

bb2:                                              ; preds = %bb1
  store i32 1, i32* %scevgep4
  br label %bb3

bb3:                                              ; preds = %bb2
  %indvar.next2 = add i64 %indvar1, 1
  br label %bb1

bb4:                                              ; preds = %bb1
  br label %bb5

bb5:                                              ; preds = %bb7, %bb4
  %indvar = phi i64 [ %indvar.next, %bb7 ], [ 0, %bb4 ]
  %tmp = add i64 %indvar, 512
  %scevgep = getelementptr [1024 x i32]* @A, i64 0, i64 %tmp
  %exitcond = icmp ne i64 %indvar, 512
  br i1 %exitcond, label %bb6, label %bb8

bb6:                                              ; preds = %bb5
  store i32 2, i32* %scevgep
  br label %bb7

bb7:                                              ; preds = %bb6
  %indvar.next = add i64 %indvar, 1
  br label %bb5

bb8:                                              ; preds = %bb5
  ret void
}

define i32 @main() nounwind {
bb:
  call void @llvm.memset.p0i8.i64(i8* bitcast ([1024 x i32]* @A to i8*), i8 0, i64 4096, i32 1, i1 false)
  call void @sequential_loops()
  br label %bb1

bb1:                                              ; preds = %bb15, %bb
  %indvar = phi i64 [ %indvar.next, %bb15 ], [ 0, %bb ]
  %i.0 = trunc i64 %indvar to i32
  %scevgep = getelementptr [1024 x i32]* @A, i64 0, i64 %indvar
  %tmp = icmp slt i32 %i.0, 1024
  br i1 %tmp, label %bb2, label %bb16

bb2:                                              ; preds = %bb1
  %tmp3 = load i32* %scevgep
  %tmp4 = icmp ne i32 %tmp3, 1
  br i1 %tmp4, label %bb5, label %bb8

bb5:                                              ; preds = %bb2
  %tmp6 = icmp slt i32 %i.0, 512
  br i1 %tmp6, label %bb7, label %bb8

bb7:                                              ; preds = %bb5
  br label %bb17

bb8:                                              ; preds = %bb5, %bb2
  %tmp9 = load i32* %scevgep
  %tmp10 = icmp ne i32 %tmp9, 2
  br i1 %tmp10, label %bb11, label %bb14

bb11:                                             ; preds = %bb8
  %tmp12 = icmp sge i32 %i.0, 512
  br i1 %tmp12, label %bb13, label %bb14

bb13:                                             ; preds = %bb11
  br label %bb17

bb14:                                             ; preds = %bb11, %bb8
  br label %bb15

bb15:                                             ; preds = %bb14
  %indvar.next = add i64 %indvar, 1
  br label %bb1

bb16:                                             ; preds = %bb1
  br label %bb17

bb17:                                             ; preds = %bb16, %bb13, %bb7
  %.0 = phi i32 [ 1, %bb7 ], [ 1, %bb13 ], [ 0, %bb16 ]
  ret i32 %.0
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
; CHECK: for (c2=0;c2<=511;c2++) {
; CHECK:     Stmt_bb2(c2);
; CHECK: }
; CHECK: for (c2=0;c2<=511;c2++) {
; CHECK:     Stmt_bb6(c2);
; CHECK: }

