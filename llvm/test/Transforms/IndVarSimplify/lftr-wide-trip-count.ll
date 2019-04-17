; RUN: opt -S -indvars < %s | FileCheck %s

; Provide legal integer types.
target datalayout = "n8:16:32:64"


define void @test1(float* %autoc,
                   float* %data,
                   float %d, i32 %data_len, i32 %sample) nounwind {
entry:
  %sub = sub i32 %data_len, %sample
  %cmp4 = icmp eq i32 %data_len, %sample
  br i1 %cmp4, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 68719476736, %entry ]
  %temp = trunc i64 %indvars.iv to i32
  %add = add i32 %temp, %sample
  %idxprom = zext i32 %add to i64
  %arrayidx = getelementptr inbounds float, float* %data, i64 %idxprom
  %temp1 = load float, float* %arrayidx, align 4
  %mul = fmul float %temp1, %d
  %arrayidx2 = getelementptr inbounds float, float* %autoc, i64 %indvars.iv
  %temp2 = load float, float* %arrayidx2, align 4
  %add3 = fadd float %temp2, %mul
  store float %add3, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %temp3 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp ult i32 %temp3, %sub
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void

; CHECK-LABEL: @test1(

; With the given initial value for IV, it is not legal to widen
; trip count to IV size
; CHECK: %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK: %lftr.wideiv = trunc i64 %indvars.iv.next to i32
; CHECK: %exitcond = icmp ne i32 %lftr.wideiv, %sub
; CHECK: br i1 %exitcond, label %for.body, label %for.end.loopexit
}

define float @test2(float* %a,
                    float* %b,
                    i32 zeroext %m) local_unnamed_addr #0 {
entry:
  %cmp5 = icmp ugt i32 %m, 500
  br i1 %cmp5, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %sum.07 = phi float [ %add, %for.body ], [ 0.000000e+00, %for.body.preheader ]
  %i.06 = phi i32 [ %inc, %for.body ], [ 500, %for.body.preheader ]
  %idxprom = zext i32 %i.06 to i64
  %arrayidx = getelementptr inbounds float, float* %b, i64 %idxprom
  %temp = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %idxprom
  %temp1 = load float, float* %arrayidx2, align 4
  %mul = fmul float %temp, %temp1
  %add = fadd float %sum.07, %mul
  %inc = add i32 %i.06, 1
  %cmp = icmp ult i32 %inc, %m
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add, %for.end.loopexit ]
  ret float %sum.0.lcssa

; CHECK-LABEL: @test2(
; Trip count should be widened and LFTR should canonicalize the condition
; CHECK: %wide.trip.count = zext
; CHECK: %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
; CHECK: br i1 %exitcond
}

define float @test3(float* %b,
                    i32 signext %m) local_unnamed_addr #0 {
entry:
  %cmp5 = icmp sgt i32 %m, -10
  br i1 %cmp5, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %sum.07 = phi float [ %add1, %for.body ], [ 0.000000e+00, %for.body.preheader ]
  %i.06 = phi i32 [ %inc, %for.body ], [ -10, %for.body.preheader ]
  %add = add nsw i32 %i.06, 20
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds float, float* %b, i64 %idxprom
  %temp = load float, float* %arrayidx, align 4
  %conv = sitofp i32 %i.06 to float
  %mul = fmul float %conv, %temp
  %add1 = fadd float %sum.07, %mul
  %inc = add nsw i32 %i.06, 1
  %cmp = icmp slt i32 %inc, %m
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add1, %for.end.loopexit ]
  ret float %sum.0.lcssa

; CHECK-LABEL: @test3(
; Trip count should be widened and LFTR should canonicalize the condition
; CHECK: %wide.trip.count = sext
; CHECK: %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
; CHECK: br i1 %exitcond
}

define float @test4(float* %b,
                    i32 signext %m) local_unnamed_addr #0 {
entry:
  %cmp5 = icmp sgt i32 %m, 10
  br i1 %cmp5, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %sum.07 = phi float [ %add1, %for.body ], [ 0.000000e+00, %for.body.preheader ]
  %i.06 = phi i32 [ %inc, %for.body ], [ 10, %for.body.preheader ]
  %add = add nsw i32 %i.06, 20
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds float, float* %b, i64 %idxprom
  %temp = load float, float* %arrayidx, align 4
  %conv = sitofp i32 %i.06 to float
  %mul = fmul float %conv, %temp
  %add1 = fadd float %sum.07, %mul
  %inc = add nsw i32 %i.06, 1
  %cmp = icmp slt i32 %inc, %m
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  %add1.lcssa = phi float [ %add1, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add1.lcssa, %for.end.loopexit ]
  ret float %sum.0.lcssa

; CHECK-LABEL: @test4(
; Trip count should be widened and LFTR should canonicalize the condition
; CHECK: %wide.trip.count = zext
; CHECK: %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
; CHECK: br i1 %exitcond
}


