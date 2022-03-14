; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; TODO: We do not allow unbounded loops at the moment.
;
; CHECK-NOT: Stmt_for_body
;
;   void f(int *A, int N, int M) {
;     for (int i = M; i > N; i++)
;       A[i] = i;
;   }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

define void @f(i32* %A, i32 %N, i32 %M) {
entry:
  br label %entry.split

entry.split:
  %cmp.1 = icmp sgt i32 %M, %N
  br i1 %cmp.1, label %for.body, label %for.end

for.body:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ %M, %entry.split ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %indvars.iv
  store i32 %indvars.iv, i32* %arrayidx, align 4
  %cmp = icmp slt i32 %M, %N
  %indvars.iv.next = add i32 %indvars.iv, 1
  br i1 %cmp, label %for.cond.for.end_crit_edge, label %for.body

for.cond.for.end_crit_edge:
  br label %for.end

for.end:
  ret void
}
