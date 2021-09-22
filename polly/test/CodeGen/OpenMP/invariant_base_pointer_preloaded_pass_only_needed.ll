; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true -polly-parallel \
; RUN: -polly-parallel-force -S < %s | FileCheck %s
;
; Test to verify that we hand down the preloaded A[0] to the OpenMP subfunction but
; not B[0] as it is not needed
;
;    void f(float *A, float *B) {
;      // Not parallel
;      for (int i = 1; i < 1000; i++) {
;        B[i] = B[i+1] + B[0];
;        // Parallel
;        for (int j = 1; j < 1000; j++)
;          A[j] += A[0];
;      }
;    }
;
;                                           i    A[0]    A
; CHECK: %polly.par.userContext = alloca { i64, float, float* }
;
; CHECK:  %polly.access.B.load =
; CHECK:  %polly.subfn.storeaddr.polly.access.A.load = getelementptr inbounds
; CHECK:  store float %polly.access.A.load, float* %polly.subfn.storeaddr.polly.access.A.load
; CHECK-NOT:  store float %polly.access.B.load, float* %polly.subfn.storeaddr.polly.access.B.load
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(float* %A, float* %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc.9, %entry
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc.9 ], [ 1, %entry ]
  %exitcond3 = icmp ne i64 %indvars.iv1, 1000
  br i1 %exitcond3, label %for.body, label %for.end.11

for.body:                                         ; preds = %for.cond
  %tmp = load float, float* %B, align 4
  %arrayidx1 = getelementptr inbounds float, float* %B, i64 %indvars.iv1
  %iv.add = add nsw i64 %indvars.iv1, 1
  %arrayidx2 = getelementptr inbounds float, float* %B, i64 %iv.add
  %tmp4 = load float, float* %arrayidx2, align 4
  %add = fadd float %tmp4, %tmp
  store float %add, float* %arrayidx1, align 4
  br label %for.cond.2

for.cond.2:                                       ; preds = %for.inc, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 1, %for.body ]
  %exitcond = icmp ne i64 %indvars.iv, 1000
  br i1 %exitcond, label %for.body.4, label %for.end

for.body.4:                                       ; preds = %for.cond.2
  %tmp5 = load float, float* %A, align 4
  %arrayidx7 = getelementptr inbounds float, float* %A, i64 %indvars.iv
  %tmp6 = load float, float* %arrayidx7, align 4
  %add8 = fadd float %tmp6, %tmp5
  store float %add8, float* %arrayidx7, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body.4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond.2

for.end:                                          ; preds = %for.cond.2
  br label %for.inc.9

for.inc.9:                                        ; preds = %for.end
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %for.cond

for.end.11:                                       ; preds = %for.cond
  ret void
}
