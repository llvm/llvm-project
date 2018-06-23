; RUN: opt < %s -loop-reroll -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i32* nocapture readonly %x) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %r.029 = phi i32 [ 0, %entry ], [ %add12, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %r.029
  %1 = or i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds i32, i32* %x, i64 %1
  %2 = load i32, i32* %arrayidx3, align 4
  %add4 = add nsw i32 %add, %2
  %3 = or i64 %indvars.iv, 2
  %arrayidx7 = getelementptr inbounds i32, i32* %x, i64 %3
  %4 = load i32, i32* %arrayidx7, align 4
  %add8 = add nsw i32 %add4, %4
  %5 = or i64 %indvars.iv, 3
  %arrayidx11 = getelementptr inbounds i32, i32* %x, i64 %5
  %6 = load i32, i32* %arrayidx11, align 4
  %add12 = add nsw i32 %add8, %6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 4
  %7 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %7, 400
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: @foo

; CHECK: for.body:
; CHECK: %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %r.029 = phi i32 [ 0, %entry ], [ %add, %for.body ]
; CHECK: %arrayidx = getelementptr inbounds i32, i32* %x, i64 %indvar
; CHECK: %1 = load i32, i32* %arrayidx, align 4
; CHECK: %add = add nsw i32 %1, %r.029
; CHECK: %indvar.next = add i64 %indvar, 1
; CHECK: %exitcond = icmp eq i32 %0, 399
; CHECK: br i1 %exitcond, label %for.end, label %for.body

; CHECK: ret

for.end:                                          ; preds = %for.body
  ret i32 %add12
}

define float @bar(float* nocapture readonly %x) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %r.029 = phi float [ 0.0, %entry ], [ %add12, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %x, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %add = fadd float %0, %r.029
  %1 = or i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds float, float* %x, i64 %1
  %2 = load float, float* %arrayidx3, align 4
  %add4 = fadd float %add, %2
  %3 = or i64 %indvars.iv, 2
  %arrayidx7 = getelementptr inbounds float, float* %x, i64 %3
  %4 = load float, float* %arrayidx7, align 4
  %add8 = fadd float %add4, %4
  %5 = or i64 %indvars.iv, 3
  %arrayidx11 = getelementptr inbounds float, float* %x, i64 %5
  %6 = load float, float* %arrayidx11, align 4
  %add12 = fadd float %add8, %6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 4
  %7 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %7, 400
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: @bar

; CHECK: for.body:
; CHECK: %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %r.029 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
; CHECK: %arrayidx = getelementptr inbounds float, float* %x, i64 %indvar
; CHECK: %1 = load float, float* %arrayidx, align 4
; CHECK: %add = fadd float %1, %r.029
; CHECK: %indvar.next = add i64 %indvar, 1
; CHECK: %exitcond = icmp eq i32 %0, 399
; CHECK: br i1 %exitcond, label %for.end, label %for.body

; CHECK: ret

for.end:                                          ; preds = %for.body
  ret float %add12
}

define i32 @foo_unusedphi(i32* nocapture readonly %x) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %r.029 = phi i32 [ 0, %entry ], [ %add12, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %0
  %1 = or i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds i32, i32* %x, i64 %1
  %2 = load i32, i32* %arrayidx3, align 4
  %add4 = add nsw i32 %add, %2
  %3 = or i64 %indvars.iv, 2
  %arrayidx7 = getelementptr inbounds i32, i32* %x, i64 %3
  %4 = load i32, i32* %arrayidx7, align 4
  %add8 = add nsw i32 %add4, %4
  %5 = or i64 %indvars.iv, 3
  %arrayidx11 = getelementptr inbounds i32, i32* %x, i64 %5
  %6 = load i32, i32* %arrayidx11, align 4
  %add12 = add nsw i32 %add8, %6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 4
  %7 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %7, 400
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: @foo_unusedphi
; The above is just testing for a crash - no specific output expected.

; CHECK: ret

for.end:                                          ; preds = %for.body
  ret i32 %add12
}

attributes #0 = { nounwind readonly uwtable }

