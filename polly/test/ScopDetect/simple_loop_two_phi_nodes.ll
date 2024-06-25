; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s

; void f(long A[], long N) {
;   long i;
;   long i_non_canonical = 1;
;   for (i = 0; i < N; ++i) {
;     A[i] = i_non_canonical;
;     i_non_canonical += 1;
;   }
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(ptr %A, i64 %N) nounwind {
entry:
  %cmp = icmp sgt i64 %N, 0
  br i1 %cmp, label %for.i, label %return

for.i:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.i ]
  %indvar_non_canonical = phi i64 [ 1, %entry ], [ %indvar_non_canonical.next, %for.i ]
  %scevgep = getelementptr i64, ptr %A, i64 %indvar
  store i64 %indvar_non_canonical, ptr %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %indvar_non_canonical.next = add nsw i64 %indvar_non_canonical, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %return, label %for.i

return:
  ret void
}

; CHECK: Valid Region for Scop: for.i => return
