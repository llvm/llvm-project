; RUN: opt < %s -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Check delinearization in loop cache analysis can handle fixed-size arrays.
; The IR is copied from llvm/test/Analysis/DependenceAnalysis/SimpleSIVNoValidityCheckFixedSize.ll

; CHECK: Loop 'for.body' has cost = 4186116
; CHECK-NEXT: Loop 'for.body4' has cost = 128898

;; #define N 1024
;; #define M 2048
;; void t1(int a[N][M]) {
;;   for (int i = 0; i < N-1; ++i)
;;     for (int j = 2; j < M; ++j)
;;       a[i][j] = a[i+1][j-2];
;; }

define void @t1([2048 x i32]* %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc11
  %indvars.iv4 = phi i64 [ 0, %entry ], [ %indvars.iv.next5, %for.inc11 ]
  br label %for.body4

for.body4:                                        ; preds = %for.body, %for.body4
  %indvars.iv = phi i64 [ 2, %for.body ], [ %indvars.iv.next, %for.body4 ]
  %0 = add nuw nsw i64 %indvars.iv4, 1
  %1 = add nsw i64 %indvars.iv, -2
  %arrayidx6 = getelementptr inbounds [2048 x i32], [2048 x i32]* %a, i64 %0, i64 %1
  %2 = load i32, i32* %arrayidx6, align 4
  %a_gep = getelementptr inbounds [2048 x i32], [2048 x i32]* %a, i64 0
  %arrayidx10 = getelementptr inbounds [2048 x i32], [2048 x i32]* %a_gep, i64 %indvars.iv4, i64 %indvars.iv
  store i32 %2, i32* %arrayidx10, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 2048
  br i1 %exitcond, label %for.body4, label %for.inc11

for.inc11:                                        ; preds = %for.body4
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 1
  %exitcond7 = icmp ne i64 %indvars.iv.next5, 1023
  br i1 %exitcond7, label %for.body, label %for.end13

for.end13:                                        ; preds = %for.inc11
  ret void
}


; CHECK: Loop 'for.body' has cost = 4186116
; CHECK-NEXT: Loop 'for.body4' has cost = 128898

define void @t2([2048 x i32]* %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc11
  %indvars.iv4 = phi i64 [ 0, %entry ], [ %indvars.iv.next5, %for.inc11 ]
  br label %for.body4

for.body4:                                        ; preds = %for.body, %for.body4
  %indvars.iv = phi i64 [ 2, %for.body ], [ %indvars.iv.next, %for.body4 ]
  %0 = add nuw nsw i64 %indvars.iv4, 1
  %1 = add nsw i64 %indvars.iv, -2
  %arrayidx6 = getelementptr inbounds [2048 x i32], [2048 x i32]* %a, i64 %0, i64 %1
  %2 = load i32, i32* %arrayidx6, align 4
  %call = call [2048 x i32]* @func_with_returned_arg([2048 x i32]* returned %a)
  %arrayidx10 = getelementptr inbounds [2048 x i32], [2048 x i32]* %call, i64 %indvars.iv4, i64 %indvars.iv
  store i32 %2, i32* %arrayidx10, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 2048
  br i1 %exitcond, label %for.body4, label %for.inc11

for.inc11:                                        ; preds = %for.body4
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 1
  %exitcond7 = icmp ne i64 %indvars.iv.next5, 1023
  br i1 %exitcond7, label %for.body, label %for.end13

for.end13:                                        ; preds = %for.inc11
  ret void
}

declare [2048 x i32]* @func_with_returned_arg([2048 x i32]* returned %arg)

; CHECK: Loop 'for.body' has cost = 2112128815104000000
; CHECK-NEXT: Loop 'for.body4' has cost = 16762927104000000
; CHECK-NEXT: Loop 'for.body8' has cost = 130960368000000
; CHECK-NEXT: Loop 'for.body12' has cost = 1047682944000
; CHECK-NEXT: Loop 'for.body16' has cost = 32260032000

;; #define N 128
;; #define M 2048
;; void t3(int a[][N][N][N][M]) {
;;   for (int i1 = 0; i1 < N-1; ++i1)
;;     for (int i2 = 2; i2 < N; ++i2)
;;       for (int i3 = 0; i3 < N; ++i3)
;;         for (int i4 = 3; i4 < N; ++i4)
;;           for (int i5 = 0; i5 < M-2; ++i5)
;;             a[i1][i2][i3][i4][i5] = a[i1+1][i2-2][i3][i4-3][i5+2];
;; }

define void @t3([128 x [128 x [128 x [2048 x i32]]]]* %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc46
  %indvars.iv18 = phi i64 [ 0, %entry ], [ %indvars.iv.next19, %for.inc46 ]
  br label %for.body4

for.body4:                                        ; preds = %for.body, %for.inc43
  %indvars.iv14 = phi i64 [ 2, %for.body ], [ %indvars.iv.next15, %for.inc43 ]
  br label %for.body8

for.body8:                                        ; preds = %for.body4, %for.inc40
  %indvars.iv11 = phi i64 [ 0, %for.body4 ], [ %indvars.iv.next12, %for.inc40 ]
  br label %for.body12

for.body12:                                       ; preds = %for.body8, %for.inc37
  %indvars.iv7 = phi i64 [ 3, %for.body8 ], [ %indvars.iv.next8, %for.inc37 ]
  br label %for.body16

for.body16:                                       ; preds = %for.body12, %for.body16
  %indvars.iv = phi i64 [ 0, %for.body12 ], [ %indvars.iv.next, %for.body16 ]
  %0 = add nuw nsw i64 %indvars.iv18, 1
  %1 = add nsw i64 %indvars.iv14, -2
  %2 = add nsw i64 %indvars.iv7, -3
  %3 = add nuw nsw i64 %indvars.iv, 2
  %arrayidx26 = getelementptr inbounds [128 x [128 x [128 x [2048 x i32]]]], [128 x [128 x [128 x [2048 x i32]]]]* %a, i64 %0, i64 %1, i64 %indvars.iv11, i64 %2, i64 %3
  %4 = load i32, i32* %arrayidx26, align 4
  %arrayidx36 = getelementptr inbounds [128 x [128 x [128 x [2048 x i32]]]], [128 x [128 x [128 x [2048 x i32]]]]* %a, i64 %indvars.iv18, i64 %indvars.iv14, i64 %indvars.iv11, i64 %indvars.iv7, i64 %indvars.iv
  store i32 %4, i32* %arrayidx36, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 2046
  br i1 %exitcond, label %for.body16, label %for.inc37

for.inc37:                                        ; preds = %for.body16
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv7, 1
  %exitcond10 = icmp ne i64 %indvars.iv.next8, 128
  br i1 %exitcond10, label %for.body12, label %for.inc40

for.inc40:                                        ; preds = %for.inc37
  %indvars.iv.next12 = add nuw nsw i64 %indvars.iv11, 1
  %exitcond13 = icmp ne i64 %indvars.iv.next12, 128
  br i1 %exitcond13, label %for.body8, label %for.inc43

for.inc43:                                        ; preds = %for.inc40
  %indvars.iv.next15 = add nuw nsw i64 %indvars.iv14, 1
  %exitcond17 = icmp ne i64 %indvars.iv.next15, 128
  br i1 %exitcond17, label %for.body4, label %for.inc46

for.inc46:                                        ; preds = %for.inc43
  %indvars.iv.next19 = add nuw nsw i64 %indvars.iv18, 1
  %exitcond21 = icmp ne i64 %indvars.iv.next19, 127
  br i1 %exitcond21, label %for.body, label %for.end48

for.end48:                                        ; preds = %for.inc46
  ret void
}
