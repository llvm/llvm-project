; RUN: opt < %s -passes='print<loopnest>' -disable-output 2>&1 | FileCheck %s

; Test that the loop nest analysis is able to analyze an infinite loop in a loop nest.
define void @test1(ptr %A, i1 %cond) {
; CHECK-LABEL: IsPerfect=true, Depth=1, OutermostLoop: for.inner, Loops: ( for.inner )
; CHECK-LABEL: IsPerfect=false, Depth=2, OutermostLoop: for.outer, Loops: ( for.outer for.inner )
; CHECK-LABEL: IsPerfect=true, Depth=1, OutermostLoop: for.infinite, Loops: ( for.infinite )
entry:
  br label %for.outer

for.outer:
  %i = phi i64 [ 0, %entry ], [ %inc_i, %for.outer.latch ]
  br i1 %cond, label %for.inner, label %for.infinite

for.inner:
  %j = phi i64 [ 0, %for.outer ], [ %inc_j, %for.inner ]
  %arrayidx_i = getelementptr inbounds ptr, ptr %A, i64 %i
  %0 = load ptr, ptr %arrayidx_i, align 8
  %arrayidx_j = getelementptr inbounds i32, ptr %0, i64 %j
  store i32 0, ptr %arrayidx_j, align 4
  %inc_j = add nsw i64 %j, 1
  %cmp_j = icmp slt i64 %inc_j, 100
  br i1 %cmp_j, label %for.inner, label %for.outer.latch

for.infinite:
  br label %for.infinite

for.outer.latch:
  %inc_i = add nsw i64 %i, 1
  %cmp_i = icmp slt i64 %inc_i, 100
  br i1 %cmp_i, label %for.outer, label %for.end

for.end:
  ret void
}
