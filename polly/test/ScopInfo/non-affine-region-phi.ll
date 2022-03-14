; RUN: opt %loadPolly -polly-allow-nonaffine -S < %s | FileCheck %s --check-prefix=CODE
; RUN: opt %loadPolly -polly-allow-nonaffine -polly-print-scops -disable-output < %s | FileCheck %s
;
; Verify there is a phi in the non-affine region but it is not represented in
; the SCoP as all operands as well as the uses are inside the region too.
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++) {
;        if (A[i]) {
;          int x = 0;
;          if (i > 512)
;            x = 1 + A[i];
;          A[i] = x;
;        }
;      }
;    }
;
; CODE-LABEL: bb11:
; CODE:         %x.0 = phi i32
;
; We have 3 accesses to A that should be present in the SCoP but no scalar access.
;
; CHECK-NOT: [Scalar: 1]
; CHECK:     [Scalar: 0]
; CHECK-NOT: [Scalar: 1]
; CHECK:     [Scalar: 0]
; CHECK-NOT: [Scalar: 1]
; CHECK:     [Scalar: 0]
; CHECK-NOT: [Scalar: 1]
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb14, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb14 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb15

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp3 = load i32,  i32* %tmp, align 4
  %tmp4 = icmp eq i32 %tmp3, 0
  br i1 %tmp4, label %bb13, label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = icmp sgt i64 %indvars.iv, 512
  br i1 %tmp6, label %bb7, label %bb11

bb7:                                              ; preds = %bb5
  %tmp8 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp9 = load i32,  i32* %tmp8, align 4
  %tmp10 = add nsw i32 %tmp9, 1
  br label %bb11

bb11:                                             ; preds = %bb7, %bb5
  %x.0 = phi i32 [ %tmp10, %bb7 ], [ 0, %bb5 ]
  %tmp12 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %x.0, i32* %tmp12, align 4
  br label %bb13

bb13:                                             ; preds = %bb2, %bb11
  br label %bb14

bb14:                                             ; preds = %bb13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb15:                                             ; preds = %bb1
  ret void
}
