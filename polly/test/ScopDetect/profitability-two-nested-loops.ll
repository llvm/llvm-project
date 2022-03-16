; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s

; CHECK: Valid Region for Scop: next => bb3
;
;    void foo(float A[], long p) {
;      for (long x = 0; x < 1024; x++) {
;        __sync_synchronize();
;        if (p >= 0) {
;          for (long i = 0; i < 1024; i++)
;            for (long j = 0; j < 1024; j++)
;              A[i + j] += j;
;        }
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define void @foo(float* %A, i64 %p) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb25, %bb
  %x.0 = phi i64 [ 0, %bb ], [ %tmp26, %bb25 ]
  %exitcond2 = icmp ne i64 %x.0, 1024
  br i1 %exitcond2, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  br label %bb27

bb5:                                              ; preds = %bb3
  fence seq_cst
  br label %next

next:
  %tmp = icmp sgt i64 %p, -1
  br i1 %tmp, label %bb6, label %bb24

bb6:                                              ; preds = %bb5
  br label %bb7

bb7:                                              ; preds = %bb21, %bb6
  %i.0 = phi i64 [ 0, %bb6 ], [ %tmp22, %bb21 ]
  %exitcond1 = icmp ne i64 %i.0, 1024
  br i1 %exitcond1, label %bb9, label %bb8

bb8:                                              ; preds = %bb7
  br label %bb23

bb9:                                              ; preds = %bb7
  br label %bb10

bb10:                                             ; preds = %bb18, %bb9
  %j.0 = phi i64 [ 0, %bb9 ], [ %tmp19, %bb18 ]
  %exitcond = icmp ne i64 %j.0, 1024
  br i1 %exitcond, label %bb12, label %bb11

bb11:                                             ; preds = %bb10
  br label %bb20

bb12:                                             ; preds = %bb10
  %tmp13 = sitofp i64 %j.0 to float
  %tmp14 = add nuw nsw i64 %i.0, %j.0
  %tmp15 = getelementptr inbounds float, float* %A, i64 %tmp14
  %tmp16 = load float, float* %tmp15, align 4
  %tmp17 = fadd float %tmp16, %tmp13
  store float %tmp17, float* %tmp15, align 4
  br label %bb18

bb18:                                             ; preds = %bb12
  %tmp19 = add nuw nsw i64 %j.0, 1
  br label %bb10

bb20:                                             ; preds = %bb11
  br label %bb21

bb21:                                             ; preds = %bb20
  %tmp22 = add nuw nsw i64 %i.0, 1
  br label %bb7

bb23:                                             ; preds = %bb8
  br label %bb24

bb24:                                             ; preds = %bb23, %bb5
  br label %bb25

bb25:                                             ; preds = %bb24
  %tmp26 = add nuw nsw i64 %x.0, 1
  br label %bb3

bb27:                                             ; preds = %bb4
  ret void
}
