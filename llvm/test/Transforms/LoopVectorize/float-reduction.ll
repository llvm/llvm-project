; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
;CHECK-LABEL: @foo(
;CHECK: fadd fast <4 x float>
;CHECK: ret
define float @foo(ptr nocapture %A, ptr nocapture %n) nounwind uwtable readonly ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.04 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %add = fadd fast float %sum.04, %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 200
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret float %add
}

;CHECK-LABEL: @foosub(
;CHECK: fsub fast <4 x float>
;CHECK: ret
define float @foosub(ptr nocapture %A, ptr nocapture %n) nounwind uwtable readonly ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.04 = phi float [ 0.000000e+00, %entry ], [ %sub, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %sub = fsub fast float %sum.04, %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 200
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret float %sub
}

;CHECK-LABEL: @foodiv(
;CHECK: fdiv fast <4 x float>
;CHECK: ret
define float @foodiv(ptr nocapture %A, ptr nocapture %n) nounwind uwtable readonly ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.04 = phi float [ 1.000000e+00, %entry ], [ %sub, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %sub = fdiv fast float %sum.04, %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 200
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret float %sub
}

;CHECK-LABEL: @foonodiv(
;CHECK-NOT: fdiv fast <4 x float>
;CHECK: ret
define float @foonodiv(ptr nocapture %A, ptr nocapture %n) nounwind uwtable readonly ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.04 = phi float [ 1.000000e+00, %entry ], [ %sub, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %sub = fdiv fast float %0, %sum.04
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 200
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret float %sub
}
