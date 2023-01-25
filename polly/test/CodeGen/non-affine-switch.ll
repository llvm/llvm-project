; RUN: opt -opaque-pointers=0 %loadPolly \
; RUN: -S -polly-codegen < %s | FileCheck %s
;
;    void f(int *A, int N) {
;      for (int i = 0; i < N; i++)
;        switch (A[i]) {
;        case 0:
;          A[i] += 1;
;          break;
;        case 1:
;          A[i] += 2;
;          break;
;        }
;    }
;
; CHECK: polly.stmt.for.body:
; CHECK:   %scevgep = getelementptr i32, i32* %A, i64 %polly.indvar
; CHECK:   %tmp1_p_scalar_ = load i32, i32* %scevgep, align 4
; CHECK:   switch i32 %tmp1_p_scalar_, label %polly.stmt.sw.epilog.exit [
; CHECK:     i32 0, label %polly.stmt.sw.bb
; CHECK:     i32 1, label %polly.stmt.sw.bb.3
; CHECK:   ]
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %N) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx, align 4
  switch i32 %tmp1, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb.3
  ]

sw.bb:                                            ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp2 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %tmp2, 1
  store i32 %add, i32* %arrayidx2, align 4
  br label %sw.epilog

sw.bb.3:                                          ; preds = %for.body
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp3 = load i32, i32* %arrayidx5, align 4
  %add6 = add nsw i32 %tmp3, 2
  store i32 %add6, i32* %arrayidx5, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb.3, %sw.bb, %for.body
  br label %for.inc

for.inc:                                          ; preds = %sw.epilog
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
