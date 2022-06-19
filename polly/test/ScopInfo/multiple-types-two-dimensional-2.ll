; RUN: opt %loadPolly -polly-print-scops -pass-remarks-analysis="polly-scops" \
; RUN:                -polly-allow-differing-element-types \
; RUN:                -disable-output < %s 2>&1 | FileCheck %s
;
;
;    void foo(long n, long m, char A[][m]) {
;      for (long i = 0; i < n; i++)
;        for (long j = 0; j < m / 4; j++)
;          *(float *)&A[i][4 * j] = A[i][j];
;    }
;
; We do not yet correctly handle multi-dimensional arrays which are accessed
; through different base types. Verify that we correctly bail out.
;
; CHECK: Delinearization assumption:  {  : false }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64 %n, i64 %m, i8* %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb20, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp21, %bb20 ]
  %tmp = icmp slt i64 %i.0, %n
  br i1 %tmp, label %bb2, label %bb22

bb2:                                              ; preds = %bb1
  br label %bb3

bb3:                                              ; preds = %bb17, %bb2
  %j.0 = phi i64 [ 0, %bb2 ], [ %tmp18, %bb17 ]
  %tmp4 = sdiv i64 %m, 4
  %tmp5 = icmp slt i64 %j.0, %tmp4
  br i1 %tmp5, label %bb6, label %bb19

bb6:                                              ; preds = %bb3
  %tmp7 = mul nsw i64 %i.0, %m
  %tmp8 = getelementptr inbounds i8, i8* %A, i64 %tmp7
  %tmp9 = getelementptr inbounds i8, i8* %tmp8, i64 %j.0
  %tmp10 = load i8, i8* %tmp9, align 1
  %tmp11 = sitofp i8 %tmp10 to float
  %tmp12 = shl nsw i64 %j.0, 2
  %tmp13 = mul nsw i64 %i.0, %m
  %tmp14 = getelementptr inbounds i8, i8* %A, i64 %tmp13
  %tmp15 = getelementptr inbounds i8, i8* %tmp14, i64 %tmp12
  %tmp16 = bitcast i8* %tmp15 to float*
  store float %tmp11, float* %tmp16, align 4
  br label %bb17

bb17:                                             ; preds = %bb6
  %tmp18 = add nuw nsw i64 %j.0, 1
  br label %bb3

bb19:                                             ; preds = %bb3
  br label %bb20

bb20:                                             ; preds = %bb19
  %tmp21 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb22:                                             ; preds = %bb1
  ret void
}
