; RUN: opt %loadPolly -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=false                                                          -polly-print-detect -disable-output < %s | FileCheck %s --check-prefix=REJECTNONAFFINELOOPS
; RUN: opt %loadPolly -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true                                                           -polly-print-detect -disable-output < %s | FileCheck %s --check-prefix=ALLOWNONAFFINELOOPS
; RUN: opt %loadPolly -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=false -polly-allow-nonaffine                                   -polly-print-detect -disable-output < %s | FileCheck %s --check-prefix=ALLOWNONAFFINEREGIONSANDACCESSES
; RUN: opt %loadPolly -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true  -polly-allow-nonaffine                                   -polly-print-detect -disable-output < %s | FileCheck %s --check-prefix=ALLOWNONAFFINELOOPSANDACCESSES
; RUN: opt %loadPolly -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true  -polly-allow-nonaffine -polly-process-unprofitable=false -polly-print-detect -disable-output < %s | FileCheck %s --check-prefix=PROFIT
;
; This function/region does contain a loop, however it is non-affine, hence the access
; A[i] is also. Furthermore, it is the only loop, thus when we over approximate
; non-affine loops __and__ accesses __and__ allow regions without a (affine) loop we will
; detect it, otherwise we won't.
;
;    void f(int *A) {
;      for (int i = 0; i < A[i]; i++)
;        A[-1]++;
;    }
;
; REJECTNONAFFINELOOPS-NOT:              Valid
; ALLOWNONAFFINELOOPS-NOT:               Valid
; ALLOWNONAFFINEREGIONSANDACCESSES-NOT:  Valid
; ALLOWNONAFFINELOOPSANDACCESSES:        Valid
; PROFIT-NOT:                            Valid
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb9, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb9 ], [ 0, %bb ]
  %tmp = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp2 = load i32, i32* %tmp, align 4
  %tmp3 = sext i32 %tmp2 to i64
  %tmp4 = icmp slt i64 %indvars.iv, %tmp3
  br i1 %tmp4, label %bb5, label %bb10

bb5:                                              ; preds = %bb1
  %tmp6 = getelementptr inbounds i32, i32* %A, i64 -1
  %tmp7 = load i32, i32* %tmp6, align 4
  %tmp8 = add nsw i32 %tmp7, 1
  store i32 %tmp8, i32* %tmp6, align 4
  br label %bb9

bb9:                                              ; preds = %bb5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb10:                                             ; preds = %bb1
  ret void
}
