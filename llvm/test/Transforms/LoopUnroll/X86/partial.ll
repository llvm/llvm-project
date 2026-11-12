; RUN: opt < %s -S -passes=loop-unroll -mcpu=nehalem | FileCheck %s
; RUN: opt < %s -S -passes=loop-unroll -unroll-runtime=0 | FileCheck -check-prefix=CHECK-NOUNRL %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr noalias nocapture readnone %ip, double %alpha, ptr noalias nocapture %a, ptr noalias nocapture readonly %b) #0 {
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds double, ptr %b, i64 %index
  %wide.load = load <2 x double>, ptr %0, align 8
  %.sum9 = or i64 %index, 2
  %1 = getelementptr double, ptr %b, i64 %.sum9
  %wide.load8 = load <2 x double>, ptr %1, align 8
  %2 = fadd <2 x double> %wide.load, <double 1.000000e+00, double 1.000000e+00>
  %3 = fadd <2 x double> %wide.load8, <double 1.000000e+00, double 1.000000e+00>
  %4 = getelementptr inbounds double, ptr %a, i64 %index
  store <2 x double> %2, ptr %4, align 8
  %.sum10 = or i64 %index, 2
  %5 = getelementptr double, ptr %a, i64 %.sum10
  store <2 x double> %3, ptr %5, align 8
  %index.next = add i64 %index, 4
  %6 = icmp eq i64 %index.next, 1600
  br i1 %6, label %for.end, label %vector.body

; FIXME: We should probably unroll this loop by a factor of 2, but the cost
; model needs to be fixed to account for instructions likely to be folded
; as part of an addressing mode.
; CHECK-LABEL: @foo
; CHECK-NOUNRL-LABEL: @foo

for.end:                                          ; preds = %vector.body
  ret void
}

define void @bar(ptr noalias nocapture readnone %ip, double %alpha, ptr noalias nocapture %a, ptr noalias nocapture readonly %b) #0 {
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %v0 = getelementptr inbounds double, ptr %b, i64 %index
  %wide.load = load <2 x double>, ptr %v0, align 8
  %v4 = fadd <2 x double> %wide.load, <double 1.000000e+00, double 1.000000e+00>
  %v5 = fmul <2 x double> %v4, <double 8.000000e+00, double 8.000000e+00>
  %v6 = getelementptr inbounds double, ptr %a, i64 %index
  store <2 x double> %v5, ptr %v6, align 8
  %index.next = add i64 %index, 2
  %v10 = icmp eq i64 %index.next, 1600
  br i1 %v10, label %for.end, label %vector.body

; FIXME: We should probably unroll this loop by a factor of 2, but the cost
; model needs to first to fixed to account for instructions likely to be folded
; as part of an addressing mode.

; CHECK-LABEL: @bar
; CHECK: fadd
; CHECK-NEXT: fmul
; CHECK: fadd
; CHECK-NEXT: fmul

; CHECK-NOUNRL-LABEL: @bar
; CHECK-NOUNRL: fadd
; CHECK-NOUNRL-NEXT: fmul
; CHECK-NOUNRL-NOT: fadd

for.end:                                          ; preds = %vector.body
  ret void
}

define zeroext i16 @test1(ptr nocapture readonly %arr, i32 %n) #0 {
entry:
  %cmp25 = icmp eq i32 %n, 0
  br i1 %cmp25, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %reduction.026 = phi i16 [ %add14, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, ptr %arr, i64 %indvars.iv
  %0 = load i16, ptr %arrayidx, align 2
  %mul = shl i16 %0, 1
  %add = add i16 %mul, %reduction.026
  %sext = mul i64 %indvars.iv, 12884901888
  %idxprom3 = ashr exact i64 %sext, 32
  %arrayidx4 = getelementptr inbounds i16, ptr %arr, i64 %idxprom3
  %1 = load i16, ptr %arrayidx4, align 2
  %mul2 = shl i16 %1, 1
  %add7 = add i16 %add, %mul2
  %sext28 = mul i64 %indvars.iv, 21474836480
  %idxprom10 = ashr exact i64 %sext28, 32
  %arrayidx11 = getelementptr inbounds i16, ptr %arr, i64 %idxprom10
  %2 = load i16, ptr %arrayidx11, align 2
  %mul3 = shl i16 %2, 1
  %add14 = add i16 %add7, %mul3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %reduction.0.lcssa = phi i16 [ 0, %entry ], [ %add14, %for.body ]
  ret i16 %reduction.0.lcssa

; This loop is too large to be partially unrolled (size=16)

; CHECK-LABEL: @test1
; CHECK: br
; CHECK: br
; CHECK: br
; CHECK: br
; CHECK-NOT: br

; CHECK-NOUNRL-LABEL: @test1
; CHECK-NOUNRL: br
; CHECK-NOUNRL: br
; CHECK-NOUNRL: br
; CHECK-NOUNRL: br
; CHECK-NOUNRL-NOT: br
}

attributes #0 = { nounwind uwtable }

