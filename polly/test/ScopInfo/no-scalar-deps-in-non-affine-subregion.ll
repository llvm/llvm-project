; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; Check that we do not generate any scalar dependences regarding x. It is
; defined and used on the non-affine subregion only, thus we do not need
; to represent the definition and uses in the model.
;
; CHECK:          Stmt_bb2__TO__bb11
; CHECK-NOT:        [Scalar: 1]
; CHECK-NOT:        MemRef_x
;
;    void f(int *A) {
;      int x;
;      for (int i = 0; i < 1024; i++) {
;        if (A[i]) {
;          if (i > 512)
;            x = 1;
;          else
;            x = 2;
;          A[i] = x;
;        }
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb12, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb12 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb13

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp3 = load i32,  ptr %tmp, align 4
  %tmp4 = icmp eq i32 %tmp3, 0
  br i1 %tmp4, label %bb11, label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = icmp sgt i64 %indvars.iv, 512
  br i1 %tmp6, label %bb7, label %bb8

bb7:                                              ; preds = %bb5
  br label %bb9

bb8:                                              ; preds = %bb5
  br label %bb9

bb9:                                              ; preds = %bb8, %bb7
  %x.0 = phi i32 [ 1, %bb7 ], [ 2, %bb8 ]
  %tmp10 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %x.0, ptr %tmp10, align 4
  br label %bb11

bb11:                                             ; preds = %bb2, %bb9
  br label %bb12

bb12:                                             ; preds = %bb11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb13:                                             ; preds = %bb1
  ret void
}
