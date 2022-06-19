; RUN: opt %loadPolly -polly-print-ast -polly-ast-detect-parallel -disable-output < %s | FileCheck %s
;
; The minimal dependence distance of the innermost loop should be 1 instead of 250.
; CHECK:    #pragma minimal dependence distance: 1
; CHECK:    for (int c0 = 0; c0 <= 499; c0 += 1)
; CHECK:      #pragma minimal dependence distance: 1
; CHECK:      for (int c1 = 0; c1 <= 998; c1 += 1) {
; CHECK:        Stmt_bb9(c0, c1);
; CHECK:        Stmt_bb9_b(c0, c1);
;
;    void foo (int *A, int *B) {
;      for (int i=0; i < 500; i++) {
;        for (int j=0; j < 1000; j++) {
;          B[i] = B[i] + 1;
;          A[j] += A[j % 250];
;        }
;      }
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

define void @foo(i32* nocapture %arg, i32* nocapture %arg1) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb
  br label %bb4

bb3:                                              ; preds = %bb6
  ret void

bb4:                                              ; preds = %bb6, %bb2
  %tmp = phi i32 [ 0, %bb2 ], [ %tmp7, %bb6 ]
  %tmp5 = getelementptr inbounds i32, i32* %arg1, i32 %tmp
  br label %bb9

bb6:                                              ; preds = %bb9
  %tmp7 = add nuw nsw i32 %tmp, 1
  %tmp8 = icmp eq i32 %tmp7, 500
  br i1 %tmp8, label %bb3, label %bb4

bb9:                                              ; preds = %bb9, %bb4
  %tmp10 = phi i32 [ 1, %bb4 ], [ %tmp19, %bb9 ]
  %tmp11 = load i32, i32* %tmp5, align 4
  %tmp12 = add nsw i32 %tmp11, 1
  store i32 %tmp12, i32* %tmp5, align 4
  %tmp13 = urem i32 %tmp10, 250
  %tmp14 = getelementptr inbounds i32, i32* %arg, i32 %tmp13
  %tmp15 = load i32, i32* %tmp14, align 4
  %tmp16 = getelementptr inbounds i32, i32* %arg, i32 %tmp10
  %tmp17 = load i32, i32* %tmp16, align 4
  %tmp18 = add nsw i32 %tmp17, %tmp15
  store i32 %tmp18, i32* %tmp16, align 4
  %tmp19 = add nuw nsw i32 %tmp10, 1
  %tmp20 = icmp eq i32 %tmp19, 1000
  br i1 %tmp20, label %bb6, label %bb9
}
