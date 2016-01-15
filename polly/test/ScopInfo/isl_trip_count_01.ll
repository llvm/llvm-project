; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; CHECK: [M, N] -> { Stmt_while_body[i0] : i0 > 0 and 4i0 <= -M + N; Stmt_while_body[0] };
;
;   void f(int *A, int N, int M) {
;     int i = 0;
;     while (M <= N) {
;       A[i++] = 1;
;       M += 4;
;     }
;   }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

define void @f(i32* nocapture %A, i32 %N, i32 %M) {
entry:
  %cmp3 = icmp sgt i32 %M, %N
  br i1 %cmp3, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %i.05 = phi i32 [ %inc, %while.body ], [ 0, %while.body.preheader ]
  %M.addr.04 = phi i32 [ %add, %while.body ], [ %M, %while.body.preheader ]
  %inc = add nuw nsw i32 %i.05, 1
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.05
  store i32 1, i32* %arrayidx, align 4
  %add = add nsw i32 %M.addr.04, 4
  %cmp = icmp sgt i32 %add, %N
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}
