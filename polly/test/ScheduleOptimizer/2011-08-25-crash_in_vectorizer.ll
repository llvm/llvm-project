; RUN: opt %loadPolly -polly-opt-isl -polly-cloog -analyze < %s -S | FileCheck %s
; RUN: opt %loadPolly -polly-opt-isl -polly-cloog -analyze %vector-opt < %s -S | FileCheck %s -check-prefix=VECTOR


target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

define void @f(i32* nocapture %A, i32 %N, i32 %C) nounwind {
bb:
  %tmp1 = icmp sgt i32 %N, 0
  br i1 %tmp1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %bb
  %tmp = zext i32 %N to i64
  br label %bb2

bb2:                                              ; preds = %bb2, %.lr.ph
  %indvar = phi i64 [ 0, %.lr.ph ], [ %indvar.next, %bb2 ]
  %scevgep = getelementptr i32* %A, i64 %indvar
  %tmp3 = load i32* %scevgep, align 4
  %tmp4 = add nsw i32 %tmp3, %C
  store i32 %tmp4, i32* %scevgep, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %tmp
  br i1 %exitcond, label %._crit_edge, label %bb2

._crit_edge:                                      ; preds = %bb2, %bb
  ret void
}

; CHECK: if (p_0 >= 1) {
; CHECK:     for (c1=0;c1<=p_0-1;c1++) {
; CHECK:         Stmt_bb2(c1);
; CHECK:     }
; CHECK: }

; VECTOR: if (p_0 >= 1) {
; VECTOR:   for (c1=0;c1<=p_0-1;c1+=4) {
; VECTOR:     for (c2=c1;c2<=min(c1+3,p_0-1);c2++) {
; VECTOR:       Stmt_bb2(c2);
; VECTOR:     }
; VECTOR:   }
; VECTOR: }
