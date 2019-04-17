; RUN: opt -loop-vectorize -S %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i128 %p1) {
entry:
  br label %while.body

while.body:
  %p.05 = phi i8* [ %add.ptr, %while.body ], [ null, %entry ]
  %p1.addr.04 = phi i128 [ %sub, %while.body ], [ %p1, %entry ]
  %add.ptr = getelementptr inbounds i8, i8* %p.05, i32 2
  %sub = add nsw i128 %p1.addr.04, -2
  %tobool = icmp eq i128 %sub, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:
  ret void
}
