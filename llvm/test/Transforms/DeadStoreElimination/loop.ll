; RUN: opt < %s -basicaa -dse -enable-nonlocal-dse -S | FileCheck %s

; The store in for.body block should be removed by non-local DSE.
; CHECK-NOT: store i32 0, i32* %arrayidx
;
define void @sum(i32 %N, i32* noalias nocapture %C, i32* noalias nocapture readonly %A, i32* noalias nocapture readonly %B) {
entry:
  %cmp24 = icmp eq i32 %N, 0
  br i1 %cmp24, label %for.end11, label %for.body

for.body:
  %i.025 = phi i32 [ %inc10, %for.cond1.for.inc9_crit_edge ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %C, i32 %i.025
  store i32 0, i32* %arrayidx, align 4
  %mul = mul i32 %i.025, %N
  %arrayidx4.gep = getelementptr i32, i32* %A, i32 %mul
  br label %for.body3

for.body3:
  %0 = phi i32 [ 0, %for.body ], [ %add8, %for.body3 ]
  %arrayidx4.phi = phi i32* [ %arrayidx4.gep, %for.body ], [ %arrayidx4.inc, %for.body3 ]
  %arrayidx5.phi = phi i32* [ %B, %for.body ], [ %arrayidx5.inc, %for.body3 ]
  %j.023 = phi i32 [ 0, %for.body ], [ %inc, %for.body3 ]
  %1 = load i32, i32* %arrayidx4.phi, align 4
  %2 = load i32, i32* %arrayidx5.phi, align 4
  %add6 = add nsw i32 %2, %1
  %add8 = add nsw i32 %add6, %0
  %inc = add i32 %j.023, 1
  %exitcond = icmp ne i32 %inc, %N
  %arrayidx4.inc = getelementptr i32, i32* %arrayidx4.phi, i32 1
  %arrayidx5.inc = getelementptr i32, i32* %arrayidx5.phi, i32 1
  br i1 %exitcond, label %for.body3, label %for.cond1.for.inc9_crit_edge

for.cond1.for.inc9_crit_edge:
  store i32 %add8, i32* %arrayidx, align 4
  %inc10 = add i32 %i.025, 1
  %exitcond26 = icmp ne i32 %inc10, %N
  br i1 %exitcond26, label %for.body, label %for.end11

for.end11:
  ret void
}
