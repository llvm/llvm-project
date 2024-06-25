; RUN: opt %loadNPMPolly  -passes=polly-codegen -polly-invariant-load-hoisting=true -polly-ignore-aliasing -polly-process-unprofitable -S < %s | FileCheck %s
;
; CHECK-LABEL: polly.preload.begin:
; CHECK-NEXT:    %0 = sext i32 %N to i64
; CHECK-NEXT:    %1 = icmp sge i64 %0, 514
; CHECK-NEXT:    %polly.preload.cond.result = and i1 %1, true
; CHECK-NEXT:    br label %polly.preload.cond
;
; CHECK-LABEL: polly.preload.cond:
; CHECK-NEXT:    br i1 %polly.preload.cond.result, label %polly.preload.exec, label %polly.preload.merge
;
; CHECK-LABEL: polly.preload.merge:
; CHECK-NEXT:    %polly.preload.tmp6.merge = phi ptr [ %polly.access.BPLoc.load, %polly.preload.exec ], [ null, %polly.preload.cond ]
;
; CHECK-LABEL: polly.stmt.bb5:
; CHECK-NEXT:    %[[offset:.*]] = shl nuw nsw i64 %polly.indvar6, 2
; CHECK-NEXT:    %{{.*}} = getelementptr i8, ptr %polly.preload.tmp6.merge, i64 %[[offset]]
;
;    void f(int **BPLoc, int *A, int N) {
;      for (int i = 0; i < N; i++)
;        if (i > 512)
;          (*BPLoc)[i] = 0;
;        else
;          A[i] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %BPLoc, ptr %A, i32 %N) {
bb:
  %tmp = sext i32 %N to i64
  br label %bb1

bb1:                                              ; preds = %bb11, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb11 ], [ 0, %bb ]
  %tmp2 = icmp slt i64 %indvars.iv, %tmp
  br i1 %tmp2, label %bb3, label %bb12

bb3:                                              ; preds = %bb1
  %tmp4 = icmp sgt i64 %indvars.iv, 512
  br i1 %tmp4, label %bb5, label %bb8

bb5:                                              ; preds = %bb3
  %tmp6 = load ptr, ptr %BPLoc, align 8
  %tmp7 = getelementptr inbounds i32, ptr %tmp6, i64 %indvars.iv
  store i32 0, ptr %tmp7, align 4
  br label %bb10

bb8:                                              ; preds = %bb3
  %tmp9 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 0, ptr %tmp9, align 4
  br label %bb10

bb10:                                             ; preds = %bb8, %bb5
  br label %bb11

bb11:                                             ; preds = %bb10
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb12:                                             ; preds = %bb1
  ret void
}
