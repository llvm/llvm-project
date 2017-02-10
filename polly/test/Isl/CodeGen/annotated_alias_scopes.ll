; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s --check-prefix=SCOPES
;
; Check that we create alias scopes that indicate the accesses to A, B and C cannot alias in any way.
;
; SCOPES-LABEL: polly.stmt.for.body:
; SCOPES:      %[[BIdx:[._a-zA-Z0-9]*]] = getelementptr{{.*}} i32* %B, i64 %polly.indvar
; SCOPES:      load i32, i32* %[[BIdx]], align 4, !alias.scope ![[AliasScopeB:[0-9]*]], !noalias ![[NoAliasB:[0-9]*]]
; SCOPES:      %[[CIdx:[._a-zA-Z0-9]*]] = getelementptr{{.*}} float* %C, i64 %polly.indvar
; SCOPES:      load float, float* %[[CIdx]], align 4, !alias.scope ![[AliasScopeC:[0-9]*]], !noalias ![[NoAliasC:[0-9]*]]
; SCOPES:      %[[AIdx:[._a-zA-Z0-9]*]] = getelementptr{{.*}} i32* %A, i64 %polly.indvar
; SCOPES:      store i32 %{{[._a-zA-Z0-9]*}}, i32* %[[AIdx]], align 4, !alias.scope ![[AliasScopeA:[0-9]*]], !noalias ![[NoAliasA:[0-9]*]]
;
; SCOPES:      ![[AliasScopeB]] = distinct !{![[AliasScopeB]], !{{[0-9]*}}, !"polly.alias.scope.MemRef_B"}
; SCOPES:      ![[NoAliasB]] = !{
; SCOPES-DAG:     ![[AliasScopeA]]
; SCOPES-DAG:     ![[AliasScopeC]]
; SCOPES:       }
; SCOPES-DAG:  ![[AliasScopeA]] = distinct !{![[AliasScopeA]], !{{[0-9]*}}, !"polly.alias.scope.MemRef_A"}
; SCOPES-DAG:  ![[AliasScopeC]] = distinct !{![[AliasScopeC]], !{{[0-9]*}}, !"polly.alias.scope.MemRef_C"}
; SCOPES:      ![[NoAliasC]] = !{
; SCOPES-DAG:     ![[AliasScopeA]]
; SCOPES-DAG:     ![[AliasScopeB]]
; SCOPES:       }
; SCOPES:      ![[NoAliasA]] = !{
; SCOPES-DAG:     ![[AliasScopeB]]
; SCOPES-DAG:     ![[AliasScopeC]]
; SCOPES:       }
;
;    void jd(int *A, int *B, float *C) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = B[i] + C[i];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A, i32* %B, float* %C) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx, align 4
  %conv = sitofp i32 %tmp to float
  %arrayidx2 = getelementptr inbounds float, float* %C, i64 %indvars.iv
  %tmp1 = load float, float* %arrayidx2, align 4
  %add = fadd fast float %conv, %tmp1
  %conv3 = fptosi float %add to i32
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %conv3, i32* %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
