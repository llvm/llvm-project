; RUN: opt -opaque-pointers=0 %loadPolly -polly-vectorizer=polly -polly-codegen \
; RUN:      < %s -S | FileCheck %s

; #pragma known-parallel
; for (int c0 = 0; c0 <= min(15, N - 1); c0 += 1)
;   Stmt_if_then(c0);

; CHECK: polly.stmt.if.then:                               ; preds = %polly.loop_header
; CHECK:   %p_conv = sitofp i64 %polly.indvar to float
; CHECK:   %scevgep = getelementptr float, float* %A, i64 %polly.indvar
; CHECK:   %_p_scalar_ = load float, float* %scevgep, align 4, !alias.scope !0, !noalias !3, !llvm.access.group !4
; CHECK:   %p_add = fadd float %p_conv, %_p_scalar_
; CHECK:   store float %p_add, float* %scevgep, align 4, !alias.scope !0, !noalias !3, !llvm.access.group !4

define void @foo(float* %A, i64 %N) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp slt i64 %i.02, %N
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %conv = sitofp i64 %i.02 to float
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.02
  %0 = load float, float* %arrayidx, align 4
  %add = fadd float %conv, %0
  store float %add, float* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 16
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}
