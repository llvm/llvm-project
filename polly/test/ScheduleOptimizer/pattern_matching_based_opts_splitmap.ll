; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-opt-isl -debug-only=polly-opt-isl -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; void pattern_matching_based_opts_splitmap(double C[static const restrict 2][2], double A[static const restrict 2][784], double B[static const restrict 784][2]) {
;  for (int i = 0; i < 2; i+=1)
;    for (int j = 0; j < 2; j+=1)
;      for (int k = 0; k < 784; k+=1)
;        C[i][j] += A[i][k] * B[k][j];
;}
;
; Check that the pattern matching detects the matrix multiplication pattern
; when the AccMap cannot be reduced to a single disjunct.
;
; CHECK: The matrix multiplication pattern was detected
;
; ModuleID = 'pattern_matching_based_opts_splitmap.ll'
;
; Function Attrs: noinline nounwind uwtable
define void @pattern_matching_based_opts_splitmap([2 x double]* noalias dereferenceable(32) %C, [784 x double]* noalias dereferenceable(12544) %A, [2 x double]* noalias dereferenceable(12544) %B) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc21
  %i = phi i64 [ 0, %entry ], [ %add22, %for.inc21 ]
  br label %for.body3

for.body3:                                        ; preds = %for.body, %for.inc18
  %j = phi i64 [ 0, %for.body ], [ %add19, %for.inc18 ]
  br label %for.body6

for.body6:                                        ; preds = %for.body3, %for.body6
  %k = phi i64 [ 0, %for.body3 ], [ %add17, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [784 x double], [784 x double]* %A, i64 %i, i64 %k
  %tmp6 = load double, double* %arrayidx8, align 8
  %arrayidx12 = getelementptr inbounds [2 x double], [2 x double]* %B, i64 %k, i64 %j
  %tmp10 = load double, double* %arrayidx12, align 8
  %mul = fmul double %tmp6, %tmp10
  %arrayidx16 = getelementptr inbounds [2 x double], [2 x double]* %C, i64 %i, i64 %j
  %tmp14 = load double, double* %arrayidx16, align 8
  %add = fadd double %tmp14, %mul
  store double %add, double* %arrayidx16, align 8
  %add17 = add nsw i64 %k, 1
  %cmp5 = icmp slt i64 %add17, 784
  br i1 %cmp5, label %for.body6, label %for.inc18

for.inc18:                                        ; preds = %for.body6
  %add19 = add nsw i64 %j, 1
  %cmp2 = icmp slt i64 %add19, 2
  br i1 %cmp2, label %for.body3, label %for.inc21

for.inc21:                                        ; preds = %for.inc18
  %add22 = add nsw i64 %i, 1
  %cmp = icmp slt i64 %add22, 2
  br i1 %cmp, label %for.body, label %for.end23

for.end23:                                        ; preds = %for.inc21
  ret void
}

