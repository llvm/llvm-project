; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

define void @foo(i32 %n, ptr nocapture %A, ptr nocapture %B) nounwind optsize {
entry:
  %cmp = icmp sgt i32 %n, 100
  br i1 %cmp, label %for.body.preheader, label %for.cond4.preheader

; CHECK: endloop0
; CHECK: endloop0
; CHECK-NOT: endloop0

for.body.preheader:
  br label %for.body

for.cond4.preheader:
  %cmp113 = icmp sgt i32 %n, 0
  br i1 %cmp113, label %for.body7.preheader, label %if.end

for.body7.preheader:
  br label %for.body7

for.body:
  %arrayidx.phi = phi ptr [ %arrayidx.inc, %for.body ], [ %B, %for.body.preheader ]
  %arrayidx3.phi = phi ptr [ %arrayidx3.inc, %for.body ], [ %A, %for.body.preheader ]
  %i.014 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %0 = load i32, ptr %arrayidx.phi, align 4
  %sub = add nsw i32 %0, -1
  store i32 %sub, ptr %arrayidx3.phi, align 4
  %inc = add nsw i32 %i.014, 1
  %exitcond = icmp eq i32 %inc, %n
  %arrayidx.inc = getelementptr i32, ptr %arrayidx.phi, i32 1
  %arrayidx3.inc = getelementptr i32, ptr %arrayidx3.phi, i32 1
  br i1 %exitcond, label %if.end.loopexit, label %for.body

for.body7:
  %arrayidx8.phi = phi ptr [ %arrayidx8.inc, %for.body7 ], [ %B, %for.body7.preheader ]
  %arrayidx9.phi = phi ptr [ %arrayidx9.inc, %for.body7 ], [ %A, %for.body7.preheader ]
  %i.117 = phi i32 [ %inc11, %for.body7 ], [ 0, %for.body7.preheader ]
  %1 = load i32, ptr %arrayidx8.phi, align 4
  %add = add nsw i32 %1, 1
  store i32 %add, ptr %arrayidx9.phi, align 4
  %inc11 = add nsw i32 %i.117, 1
  %exitcond18 = icmp eq i32 %inc11, %n
  %arrayidx8.inc = getelementptr i32, ptr %arrayidx8.phi, i32 1
  %arrayidx9.inc = getelementptr i32, ptr %arrayidx9.phi, i32 1
  br i1 %exitcond18, label %if.end.loopexit21, label %for.body7

if.end.loopexit:
  br label %if.end

if.end.loopexit21:
  br label %if.end

if.end:
  ret void
}
