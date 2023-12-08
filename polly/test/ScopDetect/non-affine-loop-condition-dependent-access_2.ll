; RUN: opt %loadPolly -basic-aa -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=false                        -polly-print-detect -disable-output < %s | FileCheck %s --check-prefix=REJECTNONAFFINELOOPS
; RUN: opt %loadPolly -basic-aa -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true                         -polly-print-detect -disable-output < %s | FileCheck %s --check-prefix=ALLOWNONAFFINELOOPS
; RUN: opt %loadPolly -basic-aa -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true  -polly-allow-nonaffine -polly-print-detect -disable-output < %s | FileCheck %s --check-prefix=ALLOWNONAFFINELOOPSANDACCESSES
;
; Here we have a non-affine loop (in the context of the loop nest)
; and also a non-affine access (A[k]). While we can always detect the
; innermost loop as a SCoP of depth 1, we have to reject the loop nest if not
; both, non-affine loops as well as non-affine accesses are allowed.
;
; REJECTNONAFFINELOOPS:           Valid Region for Scop: bb15 => bb13
; REJECTNONAFFINELOOPS-NOT:       Valid
; ALLOWNONAFFINELOOPS:            Valid Region for Scop: bb15 => bb13
; ALLOWNONAFFINELOOPS-NOT:        Valid
; ALLOWNONAFFINELOOPSANDACCESSES: Valid Region for Scop: bb11 => bb29
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
;        for (int j = 0; j < 1024; j++)
;          for (int k = i *j;  k < 1024; k++)
;            A[k] += A[i] + A[j];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A) {
bb:
  br label %bb11

bb11:                                             ; preds = %bb28, %bb
  %indvars.iv8 = phi i64 [ %indvars.iv.next9, %bb28 ], [ 0, %bb ]
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %bb28 ], [ 0, %bb ]
  %exitcond10 = icmp ne i64 %indvars.iv8, 1024
  br i1 %exitcond10, label %bb12, label %bb29

bb12:                                             ; preds = %bb11
  br label %bb13

bb13:                                             ; preds = %bb26, %bb12
  %indvars.iv5 = phi i64 [ %indvars.iv.next6, %bb26 ], [ 0, %bb12 ]
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %bb26 ], [ 0, %bb12 ]
  %exitcond7 = icmp ne i64 %indvars.iv5, 1024
  br i1 %exitcond7, label %bb14, label %bb27

bb14:                                             ; preds = %bb13
  br label %bb15

bb15:                                             ; preds = %bb24, %bb14
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb24 ], [ %indvars.iv3, %bb14 ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb16, label %bb25

bb16:                                             ; preds = %bb15
  %tmp = getelementptr inbounds i32, ptr %A, i64 %indvars.iv8
  %tmp17 = load i32, ptr %tmp, align 4
  %tmp18 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv5
  %tmp19 = load i32, ptr %tmp18, align 4
  %tmp20 = add nsw i32 %tmp17, %tmp19
  %tmp21 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp22 = load i32, ptr %tmp21, align 4
  %tmp23 = add nsw i32 %tmp22, %tmp20
  store i32 %tmp23, ptr %tmp21, align 4
  br label %bb24

bb24:                                             ; preds = %bb16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb15

bb25:                                             ; preds = %bb15
  br label %bb26

bb26:                                             ; preds = %bb25
  %indvars.iv.next6 = add nuw nsw i64 %indvars.iv5, 1
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, %indvars.iv1
  br label %bb13

bb27:                                             ; preds = %bb13
  br label %bb28

bb28:                                             ; preds = %bb27
  %indvars.iv.next9 = add nuw nsw i64 %indvars.iv8, 1
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %bb11

bb29:                                             ; preds = %bb11
  ret void
}
