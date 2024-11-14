; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s

define void @foo(i32* nocapture %A, i32 %n) {
entry:
  %n.cmp = icmp sgt i32 %n, 0
  br i1 %n.cmp, label %for.j.header, label %exit

for.j.header:
  %j= phi i32 [ %j.inc, %for.j.latch ], [ 0, %entry ]
  br label %for.i.header

for.i.header:
  %i = phi i32 [ %i.inc, %for.i.latch ], [ 0, %for.j.header ]
  br label %for.di.header

for.di.header:
  %di = phi i32 [ -1, %for.i.header ], [ %di.inc, %for.di.latch ]
  %di.add = add nsw i32 %di, %i
  br label %for.dj.header

for.dj.header:
  %dj = phi i32 [ -1, %for.di.header ], [ %dj.inc, %body ]
  %dj.add = add nsw i32 %dj, %j
  br label %while.x

while.x:
  %x = phi i32 [ %di.add, %for.dj.header ], [ %x.inc, %while.x ]
  %x.cmp = icmp slt i32 %x, 0
  %x.inc = add nsw i32 %x, %n
  br i1 %x.cmp, label %while.x, label %while.y

while.y:
  %y = phi i32 [ %y.inc, %while.y ], [ %dj.add, %while.x ]
  %y.cmp = icmp slt i32 %y, 0
  %y.inc = add nsw i32 %y, %n
  br i1 %y.cmp, label %while.y, label %body

body:
  %mul = mul nsw i32 %y, %n
  %add = add nsw i32 %mul, %x
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
; CHECK: da analyze - output [* * * *]
  store i32 7, i32* %arrayidx, align 4
  %dj.inc = add nsw i32 %dj, 1
  %dj.exitcond = icmp eq i32 %dj.inc, 2
  br i1 %dj.exitcond, label %for.di.latch, label %for.dj.header

for.di.latch:
  %di.inc = add nsw i32 %di, 1
  %di.exitcond = icmp eq i32 %di.inc, 2
  br i1 %di.exitcond, label %for.i.latch, label %for.di.header

for.i.latch:
  %i.inc = add nuw nsw i32 %i, 1
  %i.exitcond = icmp eq i32 %i.inc, %n
  br i1 %i.exitcond, label %for.j.latch, label %for.i.header

for.j.latch:
  %j.inc = add nuw nsw i32 %j, 1
  %j.exitcond = icmp eq i32 %j.inc, %n
  br i1 %j.exitcond, label %exit, label %for.j.header

exit:
  ret void
}
