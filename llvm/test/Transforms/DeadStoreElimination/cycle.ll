; RUN: opt < %s -basicaa -dse -S | FileCheck %s

@Table = global [535 x i32] zeroinitializer, align 4

; The store in for.inc block should NOT be removed by non-local DSE.
; CHECK: store i32 64, i32* %arrayidx
;
define void @foo() {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %arrayidx = getelementptr inbounds [535 x i32], [535 x i32]* @Table, i32 0, i32 %i
  store i32 %i, i32* %arrayidx, align 4
  %cmp1 = icmp slt i32 %i, 64
  br i1 %cmp1, label %for.inc, label %for.end

for.inc:
  store i32 64, i32* %arrayidx, align 4
  %inc = add nsw i32 %i, 1
  br label %for.body

for.end:
  ret void
}
