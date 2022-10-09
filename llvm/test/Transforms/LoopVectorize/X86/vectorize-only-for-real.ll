; RUN: opt -S -loop-vectorize < %s | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define i32 @accum(i32* nocapture readonly %x, i32 %N) #0 {
entry:
; CHECK-LABEL: @accum
; CHECK-NOT: x i32>

  %cmp1 = icmp sgt i32 %N, 0
  br i1 %cmp1, label %for.inc.preheader, label %for.end

for.inc.preheader:
  br label %for.inc

for.inc:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.inc.preheader ]
  %sum.02 = phi i32 [ %add, %for.inc ], [ 0, %for.inc.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.inc

for.end.loopexit:
  %add.lcssa = phi i32 [ %add, %for.inc ]
  br label %for.end

for.end:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa, %for.end.loopexit ]
  ret i32 %sum.0.lcssa

; CHECK: ret i32
}

attributes #0 = { "target-cpu"="core2" "target-features"="+sse,-avx,-avx2,-sse2" }

