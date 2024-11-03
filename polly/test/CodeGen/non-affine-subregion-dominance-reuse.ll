; RUN: opt -opaque-pointers=0 %loadPolly -polly-codegen -S -verify-dom-info \
; RUN:     < %s | FileCheck %s
;
; Check that we do not reuse the B[i-1] GEP created in block S again in
; block Q. Hence, we create two GEPs for B[i-1]:
;
; CHECK:  %scevgep{{.}} = getelementptr i32, i32* %B, i64 -1
; CHECK:  %scevgep{{.}} = getelementptr i32, i32* %B, i64 -1
;
;    void f(int *A, int *B) {
;      int x = 0;
;      for (int i = 0; i < 1024; i++) {
;        if (A[i]) {
;          if (i > 512)
; S:         A[i] = B[i - 1];
; Q:       A[i] += B[i - 1];
;        }
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32* %B) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb22, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb22 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb23

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp3 = load i32, i32* %tmp, align 4
  %tmp4 = icmp eq i32 %tmp3, 0
  br i1 %tmp4, label %bb21, label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = icmp sgt i64 %indvars.iv, 512
  br i1 %tmp6, label %bb7, label %bb13

bb7:                                              ; preds = %bb5
  br label %bb8

bb8:                                              ; preds = %bb7
  %tmp9 = add nsw i64 %indvars.iv, -1
  %tmp10 = getelementptr inbounds i32, i32* %B, i64 %tmp9
  %tmp11 = load i32, i32* %tmp10, align 4
  %tmp12 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %tmp11, i32* %tmp12, align 4
  br label %bb13

bb13:                                             ; preds = %bb8, %bb5
  br label %bb14

bb14:                                             ; preds = %bb13
  %tmp15 = add nsw i64 %indvars.iv, -1
  %tmp16 = getelementptr inbounds i32, i32* %B, i64 %tmp15
  %tmp17 = load i32, i32* %tmp16, align 4
  %tmp18 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp19 = load i32, i32* %tmp18, align 4
  %tmp20 = add nsw i32 %tmp19, %tmp17
  store i32 %tmp20, i32* %tmp18, align 4
  br label %bb21

bb21:                                             ; preds = %bb2, %bb14
  br label %bb22

bb22:                                             ; preds = %bb21
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb23:                                             ; preds = %bb1
  ret void
}
