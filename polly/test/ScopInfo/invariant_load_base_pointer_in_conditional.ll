; RUN: opt %loadPolly -polly-scops -polly-invariant-load-hoisting=true -polly-ignore-aliasing -analyze < %s | FileCheck %s
;
; CHECK: Invariant Accesses:
; CHECK-NEXT: ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:   [N] -> { Stmt_bb5[i0] -> MemRef_BP[0] };
; CHECK-NEXT:  Execution Context: [N] -> {  : N >= 514 }
;
;    void f(int *BP, int *A, int N) {
;      for (int i = 0; i < N; i++)
;        if (i > 512)
;          A[i] = *BP;
;        else
;          A[i] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %BP, i32* %A, i32 %N) {
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
  %tmp9a = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %inv = load i32, i32 *%BP
  store i32 %inv, i32* %tmp9a, align 4
  br label %bb10

bb8:                                              ; preds = %bb3
  %tmp9b = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 0, i32* %tmp9b, align 4
  br label %bb10

bb10:                                             ; preds = %bb8, %bb5
  br label %bb11

bb11:                                             ; preds = %bb10
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb12:                                             ; preds = %bb1
  ret void
}
