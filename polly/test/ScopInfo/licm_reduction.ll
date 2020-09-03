; RUN: opt %loadPolly -basic-aa -loop-rotate -indvars       -polly-prepare -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -basic-aa -loop-rotate -indvars -licm -polly-prepare -polly-scops -analyze < %s | FileCheck %s
;
; XFAIL: *
;
;    void test(int n, double B[static const restrict n], int j) {
;      for (int i = 0; i < n; i += 1) {
;        B[j] += i;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @test(i32 %n, double* noalias nonnull %B, i32 %j) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %add1, %for.inc ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %conv = sitofp i32 %i.0 to double
  %idxprom = sext i32 %j to i64
  %arrayidx = getelementptr inbounds double, double* %B, i64 %idxprom
  %tmp = load double, double* %arrayidx, align 8
  %add = fadd double %tmp, %conv
  store double %add, double* %arrayidx, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %add1 = add nuw nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}


; CHECK: Statements {
; CHECK:     Stmt_for_body
; CHECK-DAG:     ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:        [n, j] -> { Stmt_for_body[i0] -> MemRef_B[j] };
; CHECK-DAG:     MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:        [n, j] -> { Stmt_for_body[i0] -> MemRef_B[j] };
; CHECK: }
