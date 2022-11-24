; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; CHECK: Determining loop execution counts for: @test
; CHECK: Loop %for.body: backedge-taken count is ((-2 + %len) /u 2)
; CHECK: Loop %for.body: constant max backedge-taken count is 1073741823

define zeroext i16 @test(i16* nocapture %p, i32 %len) nounwind readonly {
entry:
  %cmp2 = icmp sgt i32 %len, 1
  br i1 %cmp2, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %p.addr.05 = phi i16* [ %incdec.ptr, %for.body ], [ %p, %for.body.preheader ]
  %len.addr.04 = phi i32 [ %sub, %for.body ], [ %len, %for.body.preheader ]
  %res.03 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %incdec.ptr = getelementptr inbounds i16, i16* %p.addr.05, i32 1
  %0 = load i16, i16* %p.addr.05, align 2
  %conv = zext i16 %0 to i32
  %add = add i32 %conv, %res.03
  %sub = add nsw i32 %len.addr.04, -2
  %cmp = icmp sgt i32 %sub, 1
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %extract.t = trunc i32 %add to i16
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %res.0.lcssa.off0 = phi i16 [ %extract.t, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i16 %res.0.lcssa.off0
}

