; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=false                                                          '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=REJECTNONAFFINELOOPS
; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true                                                           '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=ALLOWNONAFFINELOOPS
; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true  -polly-allow-nonaffine                                   '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=ALLOWNONAFFINELOOPSANDACCESSES
; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true  -polly-allow-nonaffine -polly-process-unprofitable=false '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=PROFIT
;
; Here we have a non-affine loop but also a non-affine access which should
; be rejected as long as -polly-allow-nonaffine isn't given.
;
; REJECTNONAFFINELOOPS-NOT:       Valid
; ALLOWNONAFFINELOOPS-NOT:        Valid
; ALLOWNONAFFINELOOPSANDACCESSES: Valid Region for Scop: bb1 => bb13
; PROFIT-NOT:                     Valid
;
;    void f(int * restrict A, int * restrict C) {
;      int j = 0;
;      for (int i = 0; i < 1024; i++) {
;        while ((j = C[j]))
;          A[j]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr noalias %A, ptr noalias %C) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb12, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb12 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb13

bb2:                                              ; preds = %bb1
  br label %bb3

bb3:                                              ; preds = %bb6, %bb2
  %indvars.j = phi i32 [ %tmp4, %bb6 ], [ 0, %bb2 ]
  %tmp = getelementptr inbounds i32, ptr %C, i32 %indvars.j
  %tmp4 = load i32, ptr %tmp, align 4
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb11, label %bb6

bb6:                                              ; preds = %bb3
  %tmp7 = sext i32 %tmp4 to i64
  %tmp8 = getelementptr inbounds i32, ptr %A, i64 %tmp7
  %tmp9 = load i32, ptr %tmp8, align 4
  %tmp10 = add nsw i32 %tmp9, 1
  store i32 %tmp10, ptr %tmp8, align 4
  br label %bb3

bb11:                                             ; preds = %bb3
  br label %bb12

bb12:                                             ; preds = %bb11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb13:                                             ; preds = %bb1
  ret void
}
