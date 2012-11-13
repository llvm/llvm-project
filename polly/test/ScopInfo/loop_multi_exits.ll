; RUN: opt %loadPolly %defaultOpts  -polly-analyze-ir  -analyze < %s | FileCheck %s  -check-prefix=INDVAR
; RUN: opt %loadPolly %defaultOpts -polly-analyze-ir  -analyze < %s | FileCheck %s
; XFAIL: *
;From pollybench.
;void f(long A[][128], long n) {
; long k, i, j;
; for (k = 0; k < n; k++) {
;   for (j = k + 1; j < n; j++)
;     A[k][j] = A[k][j] / A[k][k];
;   for(i = k + 1; i < n; i++)
;     for (j = k + 1; j < n; j++)
;       A[i][j] = A[i][j] - A[i][k] * A[k][j];
; }
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-linux-gnu"

define void @f([128 x i64]* nocapture %A, i64 %n) nounwind {
entry:
  %0 = icmp sgt i64 %n, 0                         ; <i1> [#uses=1]
  br i1 %0, label %bb.nph30, label %return

bb.nph:                                           ; preds = %bb2.preheader
  %1 = getelementptr inbounds [128 x i64]* %A, i64 %k.023, i64 %k.023 ; <i64*> [#uses=1]
  %tmp31 = sub i64 %tmp, %k.023                   ; <i64> [#uses=1]
  %tmp32 = mul i64 %k.023, 129                    ; <i64> [#uses=1]
  %tmp33 = add i64 %tmp32, 1                      ; <i64> [#uses=1]
  br label %bb1

bb1:                                              ; preds = %bb1, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb1 ] ; <i64> [#uses=2]
  %tmp34 = add i64 %tmp33, %indvar                ; <i64> [#uses=1]
  %scevgep = getelementptr [128 x i64]* %A, i64 0, i64 %tmp34 ; <i64*> [#uses=2]
  %2 = load i64* %scevgep, align 8                ; <i64> [#uses=1]
  %3 = load i64* %1, align 8                      ; <i64> [#uses=1]
  %4 = sdiv i64 %2, %3                            ; <i64> [#uses=1]
  store i64 %4, i64* %scevgep, align 8
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next, %tmp31    ; <i1> [#uses=1]
  br i1 %exitcond, label %bb8.loopexit, label %bb1

bb.nph16:                                         ; preds = %bb.nph22, %bb8.loopexit12
  %indvar39 = phi i64 [ 0, %bb.nph22 ], [ %tmp51, %bb8.loopexit12 ] ; <i64> [#uses=2]
  %tmp48 = add i64 %j.013, %indvar39              ; <i64> [#uses=1]
  %tmp51 = add i64 %indvar39, 1                   ; <i64> [#uses=3]
  %scevgep53 = getelementptr [128 x i64]* %A, i64 %tmp51, i64 %tmp52 ; <i64*> [#uses=1]
  %tmp37 = sub i64 %n, %j.013                     ; <i64> [#uses=1]
  br label %bb5

bb5:                                              ; preds = %bb5, %bb.nph16
  %indvar35 = phi i64 [ 0, %bb.nph16 ], [ %indvar.next36, %bb5 ] ; <i64> [#uses=2]
  %tmp49 = add i64 %j.013, %indvar35              ; <i64> [#uses=2]
  %scevgep43 = getelementptr [128 x i64]* %A, i64 %tmp48, i64 %tmp49 ; <i64*> [#uses=2]
  %scevgep44 = getelementptr [128 x i64]* %A, i64 %k.023, i64 %tmp49 ; <i64*> [#uses=1]
  %5 = load i64* %scevgep43, align 8              ; <i64> [#uses=1]
  %6 = load i64* %scevgep53, align 8              ; <i64> [#uses=1]
  %7 = load i64* %scevgep44, align 8              ; <i64> [#uses=1]
  %8 = mul nsw i64 %7, %6                         ; <i64> [#uses=1]
  %9 = sub nsw i64 %5, %8                         ; <i64> [#uses=1]
  store i64 %9, i64* %scevgep43, align 8
  %indvar.next36 = add i64 %indvar35, 1           ; <i64> [#uses=2]
  %exitcond38 = icmp eq i64 %indvar.next36, %tmp37 ; <i1> [#uses=1]
  br i1 %exitcond38, label %bb8.loopexit12, label %bb5

bb8.loopexit:                                     ; preds = %bb1
  br i1 %10, label %bb.nph22, label %return

bb8.loopexit12:                                   ; preds = %bb5
  %exitcond47 = icmp eq i64 %tmp51, %tmp46        ; <i1> [#uses=1]
  br i1 %exitcond47, label %bb10.loopexit, label %bb.nph16

bb.nph22:                                         ; preds = %bb8.loopexit
  %tmp46 = sub i64 %tmp, %k.023                   ; <i64> [#uses=1]
  %tmp52 = mul i64 %k.023, 129                    ; <i64> [#uses=1]
  br label %bb.nph16

bb10.loopexit:                                    ; preds = %bb8.loopexit12
  br i1 %10, label %bb2.preheader, label %return

bb.nph30:                                         ; preds = %entry
  %tmp = add i64 %n, -1                           ; <i64> [#uses=2]
  br label %bb2.preheader

bb2.preheader:                                    ; preds = %bb.nph30, %bb10.loopexit
  %k.023 = phi i64 [ 0, %bb.nph30 ], [ %j.013, %bb10.loopexit ] ; <i64> [#uses=8]
  %j.013 = add i64 %k.023, 1                      ; <i64> [#uses=5]
  %10 = icmp slt i64 %j.013, %n                   ; <i1> [#uses=3]
  br i1 %10, label %bb.nph, label %return

return:                                           ; preds = %bb2.preheader, %bb10.loopexit, %bb8.loopexit, %entry
  ret void
}

; CHECK: Scop: bb5 => bb8.loopexit12     Parameters: ({0,+,1}<%bb2.preheader>, %n, {0,+,1}<%bb.nph16>, ), Max Loop Depth: 1
; CHECK: Scop: bb.nph16 => bb10.loopexit Parameters: ({0,+,1}<%bb2.preheader>, %n, ), Max Loop Depth: 2
; CHECK: Scop: bb1 => bb8.loopexit       Parameters: ({0,+,1}<%bb2.preheader>, %n, ), Max Loop Depth: 1

; INDVAR: Scop: bb1 => bb8.loopexit       Parameters: ({0,+,1}<%bb2.preheader>, %n, ), Max Loop Depth: 1
; INDVAR: Scop: bb.nph16 => bb10.loopexit Parameters: ({0,+,1}<%bb2.preheader>, %n, ), Max Loop Depth: 2
; INDVAR: Scop: bb5 => bb8.loopexit12     Parameters: ({0,+,1}<%bb2.preheader>, %n, {0,+,1}<%bb.nph16>, ), Max Loop Depth: 1
