; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    void f(long *A, long N, long p) {
;      for (long i = 0; i < N; i++)
;        A[i + 1] = 0;
;    }
;
; The wrap function has no inbounds GEP but the nowrap function has. Therefore,
; we will add the assumption that i+1 won't overflow only to the former.
;
; Note:
;       1152921504606846975 * sizeof(long) <= 2 ^ 63 - 1
;  and
;       1152921504606846976 * sizeof(long) >  2 ^ 63 - 1
; with
;       sizeof(long) == 8
;
; CHECK:      Function: wrap
; CHECK:      Invalid Context:
; CHECK-NEXT: [N] -> {  : N >= 1152921504606846976 }
;
; CHECK:      Function: nowrap
; CHECK:      Invalid Context:
; CHECK-NEXT: [N] -> {  : false }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @wrap(i64* %A, i64 %N, i64 %p) {
bb:
  %tmp31 = icmp slt i64 0, %N
  br i1 %tmp31, label %bb4.lr.ph, label %bb8

bb4.lr.ph:                                        ; preds = %bb
  br label %bb4

bb4:                                              ; preds = %bb4.lr.ph, %bb7
  %indvars.iv2 = phi i64 [ 0, %bb4.lr.ph ], [ %indvars.iv.next, %bb7 ]
  %tmp5 = add nuw nsw i64 %indvars.iv2, 1
  %tmp6 = getelementptr i64, i64* %A, i64 %tmp5
  store i64 0, i64* %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv2, 1
  %tmp3 = icmp slt i64 %indvars.iv.next, %N
  br i1 %tmp3, label %bb4, label %bb2.bb8_crit_edge

bb2.bb8_crit_edge:                                ; preds = %bb7
  br label %bb8

bb8:                                              ; preds = %bb2.bb8_crit_edge, %bb
  ret void
}

define void @nowrap(i64* %A, i64 %N, i64 %p) {
bb:
  %tmp31 = icmp slt i64 0, %N
  br i1 %tmp31, label %bb4.lr.ph, label %bb8

bb4.lr.ph:                                        ; preds = %bb
  br label %bb4

bb4:                                              ; preds = %bb4.lr.ph, %bb7
  %indvars.iv2 = phi i64 [ 0, %bb4.lr.ph ], [ %indvars.iv.next, %bb7 ]
  %tmp5 = add nuw nsw i64 %indvars.iv2, 1
  %tmp6 = getelementptr inbounds i64, i64* %A, i64 %tmp5
  store i64 0, i64* %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv2, 1
  %tmp3 = icmp slt i64 %indvars.iv.next, %N
  br i1 %tmp3, label %bb4, label %bb2.bb8_crit_edge

bb2.bb8_crit_edge:                                ; preds = %bb7
  br label %bb8

bb8:                                              ; preds = %bb2.bb8_crit_edge, %bb
  ret void
}
