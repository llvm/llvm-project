; RUN: opt %loadNPMPolly '-passes=print<polly-detect>,print<polly-function-scops>' -polly-invariant-load-hoisting=true -disable-output < %s 2>&1 | FileCheck %s
;
; Negative test. If we assume UB[*V] to be invariant we get a cyclic
; dependence in the invariant loads that needs to be resolved by
; ignoring the actual accessed address and focusing on the fact
; that the access happened. However, at the moment we assume UB[*V]
; not to be loop invariant, thus reject this region.
;
; CHECK-NOT: Statements
;
;
;    void f(int *restrict V, int *restrict UB, int *restrict A) {
;      for (int i = 0; i < 100; i++) {
;        int j = 0;
;        int x = 0;
;        do {
;          x = /* invariant load dependent on UB[*V] */ *V;
;          A[j + i]++;
;        } while (j++ < /* invariant load dependent on *V */ UB[x]);
;      }
;    }
;
target datalayout = "e-m:e-i32:64-f80:128-n8:16:32:64-S128"

define void @f(ptr noalias %V, ptr noalias %UB, ptr noalias %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv2 = phi i32 [ %indvars.iv.next3, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i32 %indvars.iv2, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br label %do.body

do.body:                                          ; preds = %do.cond, %for.body
  %indvars.iv = phi i32 [ %indvars.iv.next, %do.cond ], [ 0, %for.body ]
  %tmp = load i32, ptr %V, align 4
  %tmp4 = add nuw nsw i32 %indvars.iv, %indvars.iv2
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %tmp4
  %tmp5 = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %tmp5, 1
  store i32 %inc, ptr %arrayidx, align 4
  br label %do.cond

do.cond:                                          ; preds = %do.body
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds i32, ptr %UB, i32 %tmp
  %tmp6 = load i32, ptr %arrayidx3, align 4
  %cmp4 = icmp slt i32 %indvars.iv, %tmp6
  br i1 %cmp4, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  br label %for.inc

for.inc:                                          ; preds = %do.end
  %indvars.iv.next3 = add nuw nsw i32 %indvars.iv2, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
