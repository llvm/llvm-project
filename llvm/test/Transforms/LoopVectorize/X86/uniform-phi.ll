; RUN: opt < %s -passes=loop-vectorize -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7 -debug-only=loop-vectorize -S 2>&1 | FileCheck %s
; REQUIRES: asserts
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: test
; CHECK-DAG: LV: Found uniform instruction:   %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK-DAG: LV: Found uniform instruction:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-DAG: LV: Found uniform instruction:   %exitcond = icmp eq i64 %indvars.iv, 1599

define void @test(ptr noalias nocapture %a, ptr noalias nocapture readonly %b) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %tmp0 = load float, ptr %arrayidx, align 4
  %add = fadd float %tmp0, 1.000000e+00
  %arrayidx5 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  store float %add, ptr %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 1599
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; CHECK-LABEL: foo
; CHECK-DAG: LV: Found uniform instruction:   %cond = icmp eq i64 %i.next, %n
; CHECK-DAG: LV: Found uniform instruction:   %tmp1 = getelementptr inbounds i32, ptr %a, i32 %tmp0
; CHECK-NOT: LV: Found uniform instruction:   %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]

define void @foo(ptr %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = trunc i64 %i to i32
  %tmp1 = getelementptr inbounds i32, ptr %a, i32 %tmp0
  store i32 %tmp0, ptr %tmp1, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, %n
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK-LABEL: goo
; Check %indvars.iv and %indvars.iv.next are uniform instructions even if they are used outside of loop.
; CHECK-DAG: LV: Found uniform instruction:   %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK-DAG: LV: Found uniform instruction:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-DAG: LV: Found uniform instruction:   %exitcond = icmp eq i64 %indvars.iv, 1599

define i64 @goo(ptr noalias nocapture %a, ptr noalias nocapture readonly %b) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %tmp0 = load float, ptr %arrayidx, align 4
  %add = fadd float %tmp0, 1.000000e+00
  %arrayidx5 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  store float %add, ptr %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 1599
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %retval = add i64 %indvars.iv, %indvars.iv.next
  ret i64 %retval
}

; CHECK-LABEL: PR38786
; Check that first order recurrence phis (%phi32 and %phi64) are not uniform.
; CHECK-NOT: LV: Found uniform instruction:   %phi
define void @PR38786(ptr %y, ptr %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %phi32 = phi i32 [ 0, %entry ], [ %i32next, %for.body ]
  %phi64 = phi i64 [ 0, %entry ], [ %i64next, %for.body ]
  %i32next = add i32 %phi32, 1
  %i64next = zext i32 %i32next to i64
  %xip = getelementptr inbounds double, ptr %x, i64 %i64next
  %yip = getelementptr inbounds double, ptr %y, i64 %phi64
  %xi = load double, ptr %xip, align 8
  store double %xi, ptr %yip, align 8
  %cmp = icmp slt i64 %i64next, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}
