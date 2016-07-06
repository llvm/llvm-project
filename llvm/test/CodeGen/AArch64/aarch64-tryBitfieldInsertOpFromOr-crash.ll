; RUN: llc <%s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; Function Attrs: noreturn nounwind
define void @foo(i32* %d) {
entry:
  %0 = ptrtoint i32* %d to i64
  %1 = and i64 %0, -36028797018963969
  %2 = inttoptr i64 %1 to i32*
  %arrayidx5 = getelementptr inbounds i32, i32* %2, i64 1
  %arrayidx6 = getelementptr inbounds i32, i32* %2, i64 2
  %arrayidx7 = getelementptr inbounds i32, i32* %2, i64 3
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %B.0 = phi i32* [ %d, %entry ], [ %12, %for.cond ]
  %3 = ptrtoint i32* %B.0 to i64
  %4 = and i64 %3, -36028797018963969
  %5 = inttoptr i64 %4 to i32*
  %6 = load i32, i32* %5, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %5, i64 1
  %7 = load i32, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %5, i64 2
  %8 = load i32, i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %5, i64 3
  %9 = load i32, i32* %arrayidx3, align 4
  store i32 %6, i32* %2, align 4
  store i32 %7, i32* %arrayidx5, align 4
  store i32 %8, i32* %arrayidx6, align 4
  store i32 %9, i32* %arrayidx7, align 4
  %10 = ptrtoint i32* %arrayidx1 to i64
  %11 = or i64 %10, 36028797018963968
  %12 = inttoptr i64 %11 to i32*
  br label %for.cond
}
