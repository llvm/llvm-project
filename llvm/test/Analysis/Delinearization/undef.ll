; RUN: opt < %s -passes='print<delinearization>' -disable-output
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %Ey) {
entry:
  br i1 undef, label %for.cond55.preheader, label %for.end324

for.cond55.preheader:
  %iz.069 = phi i64 [ %inc323, %for.inc322 ], [ 0, %entry ]
  br i1 undef, label %for.cond58.preheader, label %for.inc322

for.cond58.preheader:
  %iy.067 = phi i64 [ %inc320, %for.end ], [ 0, %for.cond55.preheader ]
  br i1 undef, label %for.body60, label %for.end

for.body60:
  %ix.062 = phi i64 [ %inc, %for.body60 ], [ 0, %for.cond58.preheader ]
  %0 = mul i64 %iz.069, undef
  %tmp5 = add i64 %iy.067, %0
  %tmp6 = mul i64 %tmp5, undef
  %arrayidx69.sum = add i64 undef, %tmp6
  %arrayidx70 = getelementptr inbounds double, ptr %Ey, i64 %arrayidx69.sum
  %1 = load double, ptr %arrayidx70, align 8
  %inc = add nsw i64 %ix.062, 1
  br i1 false, label %for.body60, label %for.end

for.end:
  %inc320 = add nsw i64 %iy.067, 1
  br i1 undef, label %for.cond58.preheader, label %for.inc322

for.inc322:
  %inc323 = add nsw i64 %iz.069, 1
  br i1 undef, label %for.cond55.preheader, label %for.end324

for.end324:
  ret void
}
