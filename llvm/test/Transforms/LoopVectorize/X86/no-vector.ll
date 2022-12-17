; RUN: opt -S -mtriple=i386-unknown-freebsd -mcpu=i486 -passes=loop-vectorize < %s

define i32 @PR14639(i8* nocapture %s, i32 %len) nounwind {
entry:
  %cmp4 = icmp sgt i32 %len, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %r.05 = phi i32 [ %xor, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %s, i32 %i.06
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %xor = xor i32 %conv, %r.05
  %inc = add nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, %len
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %r.0.lcssa = phi i32 [ 0, %entry ], [ %xor, %for.body ]
  ret i32 %r.0.lcssa
}
