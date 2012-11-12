; RUN: opt -S -indvars < %s | FileCheck %s

define void @test1(float* nocapture %autoc, float* nocapture %data, float %d, i32 %data_len, i32 %sample) nounwind {
entry:
  %sub = sub i32 %data_len, %sample
  %cmp4 = icmp eq i32 %data_len, %sample
  br i1 %cmp4, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %0 = trunc i64 %indvars.iv to i32
  %add = add i32 %0, %sample
  %idxprom = zext i32 %add to i64
  %arrayidx = getelementptr inbounds float* %data, i64 %idxprom
  %1 = load float* %arrayidx, align 4
  %mul = fmul float %1, %d
  %arrayidx2 = getelementptr inbounds float* %autoc, i64 %indvars.iv
  %2 = load float* %arrayidx2, align 4
  %add3 = fadd float %2, %mul
  store float %add3, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %3 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp ult i32 %3, %sub
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void

; CHECK: @test1

; First check that we move the sub into the preheader, it doesn't have to be
; executed if %cmp4 == false
; CHECK: for.body.preheader:
; CHECK: sub i32 %data_len, %sample
; CHECK: br label %for.body

; Second, check that we turn the IV test into an eq.
; CHECK: %lftr.wideiv = trunc i64 %indvars.iv.next to i32
; CHECK: %exitcond = icmp ne i32 %lftr.wideiv, %0
; CHECK: br i1 %exitcond, label %for.body, label %for.end.loopexit
}

