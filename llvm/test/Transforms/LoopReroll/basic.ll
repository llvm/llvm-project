; RUN: opt < %s -loop-reroll -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; int foo(int a);
; void bar(int *x) {
;   for (int i = 0; i < 500; i += 3) {
;     foo(i);
;     foo(i+1);
;     foo(i+2);
;   }
; }

; Function Attrs: nounwind uwtable
define void @bar(i32* nocapture readnone %x) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %call = tail call i32 @foo(i32 %i.08) #1
  %add = add nsw i32 %i.08, 1
  %call1 = tail call i32 @foo(i32 %add) #1
  %add2 = add nsw i32 %i.08, 2
  %call3 = tail call i32 @foo(i32 %add2) #1
  %add3 = add nsw i32 %i.08, 3
  %exitcond = icmp eq i32 %add3, 500
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: @bar

; CHECK: for.body:
; CHECK: %indvar = phi i32 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %call = tail call i32 @foo(i32 %indvar) #1
; CHECK: %indvar.next = add i32 %indvar, 1
; CHECK: %exitcond1 = icmp eq i32 %indvar, 497
; CHECK: br i1 %exitcond1, label %for.end, label %for.body

; CHECK: ret

for.end:                                          ; preds = %for.body
  ret void
}

declare i32 @foo(i32)

; void hi1(int *x) {
;   for (int i = 0; i < 1500; i += 3) {
;     x[i] = foo(0);
;     x[i+1] = foo(0);
;     x[i+2] = foo(0);
;   }
; }

; Function Attrs: nounwind uwtable
define void @hi1(i32* nocapture %x) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %call = tail call i32 @foo(i32 0) #1
  %arrayidx = getelementptr inbounds i32* %x, i64 %indvars.iv
  store i32 %call, i32* %arrayidx, align 4
  %call1 = tail call i32 @foo(i32 0) #1
  %0 = add nsw i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds i32* %x, i64 %0
  store i32 %call1, i32* %arrayidx3, align 4
  %call4 = tail call i32 @foo(i32 0) #1
  %1 = add nsw i64 %indvars.iv, 2
  %arrayidx7 = getelementptr inbounds i32* %x, i64 %1
  store i32 %call4, i32* %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 3
  %2 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %2, 1500
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: @hi1

; CHECK: for.body:
; CHECK: %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %call = tail call i32 @foo(i32 0) #1
; CHECK: %arrayidx = getelementptr inbounds i32* %x, i64 %indvar
; CHECK: store i32 %call, i32* %arrayidx, align 4
; CHECK: %indvar.next = add i64 %indvar, 1
; CHECK: %exitcond = icmp eq i64 %indvar, 1499
; CHECK: br i1 %exitcond, label %for.end, label %for.body

; CHECK: ret

for.end:                                          ; preds = %for.body
  ret void
}

; void hi2(int *x) {
;   for (int i = 0; i < 500; ++i) {
;     x[3*i] = foo(0);
;     x[3*i+1] = foo(0);
;     x[3*i+2] = foo(0);
;   }
; }

; Function Attrs: nounwind uwtable
define void @hi2(i32* nocapture %x) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %call = tail call i32 @foo(i32 0) #1
  %0 = mul nsw i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds i32* %x, i64 %0
  store i32 %call, i32* %arrayidx, align 4
  %call1 = tail call i32 @foo(i32 0) #1
  %1 = add nsw i64 %0, 1
  %arrayidx4 = getelementptr inbounds i32* %x, i64 %1
  store i32 %call1, i32* %arrayidx4, align 4
  %call5 = tail call i32 @foo(i32 0) #1
  %2 = add nsw i64 %0, 2
  %arrayidx9 = getelementptr inbounds i32* %x, i64 %2
  store i32 %call5, i32* %arrayidx9, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 500
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: @hi2

; CHECK: for.body:
; CHECK: %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK: %call = tail call i32 @foo(i32 0) #1
; CHECK: %arrayidx = getelementptr inbounds i32* %x, i64 %indvars.iv
; CHECK: store i32 %call, i32* %arrayidx, align 4
; CHECK: %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK: %exitcond1 = icmp eq i64 %indvars.iv, 1499
; CHECK: br i1 %exitcond1, label %for.end, label %for.body

; CHECK: ret

for.end:                                          ; preds = %for.body
  ret void
}

; void goo(float alpha, float *a, float *b) {
;   for (int i = 0; i < 3200; i += 5) {
;     a[i] += alpha * b[i];
;     a[i + 1] += alpha * b[i + 1];
;     a[i + 2] += alpha * b[i + 2];
;     a[i + 3] += alpha * b[i + 3];
;     a[i + 4] += alpha * b[i + 4];
;   }
; }

; Function Attrs: nounwind uwtable
define void @goo(float %alpha, float* nocapture %a, float* nocapture readonly %b) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float* %b, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4
  %mul = fmul float %0, %alpha
  %arrayidx2 = getelementptr inbounds float* %a, i64 %indvars.iv
  %1 = load float* %arrayidx2, align 4
  %add = fadd float %1, %mul
  store float %add, float* %arrayidx2, align 4
  %2 = add nsw i64 %indvars.iv, 1
  %arrayidx5 = getelementptr inbounds float* %b, i64 %2
  %3 = load float* %arrayidx5, align 4
  %mul6 = fmul float %3, %alpha
  %arrayidx9 = getelementptr inbounds float* %a, i64 %2
  %4 = load float* %arrayidx9, align 4
  %add10 = fadd float %4, %mul6
  store float %add10, float* %arrayidx9, align 4
  %5 = add nsw i64 %indvars.iv, 2
  %arrayidx13 = getelementptr inbounds float* %b, i64 %5
  %6 = load float* %arrayidx13, align 4
  %mul14 = fmul float %6, %alpha
  %arrayidx17 = getelementptr inbounds float* %a, i64 %5
  %7 = load float* %arrayidx17, align 4
  %add18 = fadd float %7, %mul14
  store float %add18, float* %arrayidx17, align 4
  %8 = add nsw i64 %indvars.iv, 3
  %arrayidx21 = getelementptr inbounds float* %b, i64 %8
  %9 = load float* %arrayidx21, align 4
  %mul22 = fmul float %9, %alpha
  %arrayidx25 = getelementptr inbounds float* %a, i64 %8
  %10 = load float* %arrayidx25, align 4
  %add26 = fadd float %10, %mul22
  store float %add26, float* %arrayidx25, align 4
  %11 = add nsw i64 %indvars.iv, 4
  %arrayidx29 = getelementptr inbounds float* %b, i64 %11
  %12 = load float* %arrayidx29, align 4
  %mul30 = fmul float %12, %alpha
  %arrayidx33 = getelementptr inbounds float* %a, i64 %11
  %13 = load float* %arrayidx33, align 4
  %add34 = fadd float %13, %mul30
  store float %add34, float* %arrayidx33, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 5
  %14 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %14, 3200
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: @goo

; CHECK: for.body:
; CHECK: %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %arrayidx = getelementptr inbounds float* %b, i64 %indvar
; CHECK: %0 = load float* %arrayidx, align 4
; CHECK: %mul = fmul float %0, %alpha
; CHECK: %arrayidx2 = getelementptr inbounds float* %a, i64 %indvar
; CHECK: %1 = load float* %arrayidx2, align 4
; CHECK: %add = fadd float %1, %mul
; CHECK: store float %add, float* %arrayidx2, align 4
; CHECK: %indvar.next = add i64 %indvar, 1
; CHECK: %exitcond = icmp eq i64 %indvar, 3199
; CHECK: br i1 %exitcond, label %for.end, label %for.body

; CHECK: ret

for.end:                                          ; preds = %for.body
  ret void
}

; void hoo(float alpha, float *a, float *b, int *ip) {
;   for (int i = 0; i < 3200; i += 5) {
;     a[i] += alpha * b[ip[i]];
;     a[i + 1] += alpha * b[ip[i + 1]];
;     a[i + 2] += alpha * b[ip[i + 2]];
;     a[i + 3] += alpha * b[ip[i + 3]];
;     a[i + 4] += alpha * b[ip[i + 4]];
;   }
; }

; Function Attrs: nounwind uwtable
define void @hoo(float %alpha, float* nocapture %a, float* nocapture readonly %b, i32* nocapture readonly %ip) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32* %ip, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %idxprom1 = sext i32 %0 to i64
  %arrayidx2 = getelementptr inbounds float* %b, i64 %idxprom1
  %1 = load float* %arrayidx2, align 4
  %mul = fmul float %1, %alpha
  %arrayidx4 = getelementptr inbounds float* %a, i64 %indvars.iv
  %2 = load float* %arrayidx4, align 4
  %add = fadd float %2, %mul
  store float %add, float* %arrayidx4, align 4
  %3 = add nsw i64 %indvars.iv, 1
  %arrayidx7 = getelementptr inbounds i32* %ip, i64 %3
  %4 = load i32* %arrayidx7, align 4
  %idxprom8 = sext i32 %4 to i64
  %arrayidx9 = getelementptr inbounds float* %b, i64 %idxprom8
  %5 = load float* %arrayidx9, align 4
  %mul10 = fmul float %5, %alpha
  %arrayidx13 = getelementptr inbounds float* %a, i64 %3
  %6 = load float* %arrayidx13, align 4
  %add14 = fadd float %6, %mul10
  store float %add14, float* %arrayidx13, align 4
  %7 = add nsw i64 %indvars.iv, 2
  %arrayidx17 = getelementptr inbounds i32* %ip, i64 %7
  %8 = load i32* %arrayidx17, align 4
  %idxprom18 = sext i32 %8 to i64
  %arrayidx19 = getelementptr inbounds float* %b, i64 %idxprom18
  %9 = load float* %arrayidx19, align 4
  %mul20 = fmul float %9, %alpha
  %arrayidx23 = getelementptr inbounds float* %a, i64 %7
  %10 = load float* %arrayidx23, align 4
  %add24 = fadd float %10, %mul20
  store float %add24, float* %arrayidx23, align 4
  %11 = add nsw i64 %indvars.iv, 3
  %arrayidx27 = getelementptr inbounds i32* %ip, i64 %11
  %12 = load i32* %arrayidx27, align 4
  %idxprom28 = sext i32 %12 to i64
  %arrayidx29 = getelementptr inbounds float* %b, i64 %idxprom28
  %13 = load float* %arrayidx29, align 4
  %mul30 = fmul float %13, %alpha
  %arrayidx33 = getelementptr inbounds float* %a, i64 %11
  %14 = load float* %arrayidx33, align 4
  %add34 = fadd float %14, %mul30
  store float %add34, float* %arrayidx33, align 4
  %15 = add nsw i64 %indvars.iv, 4
  %arrayidx37 = getelementptr inbounds i32* %ip, i64 %15
  %16 = load i32* %arrayidx37, align 4
  %idxprom38 = sext i32 %16 to i64
  %arrayidx39 = getelementptr inbounds float* %b, i64 %idxprom38
  %17 = load float* %arrayidx39, align 4
  %mul40 = fmul float %17, %alpha
  %arrayidx43 = getelementptr inbounds float* %a, i64 %15
  %18 = load float* %arrayidx43, align 4
  %add44 = fadd float %18, %mul40
  store float %add44, float* %arrayidx43, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 5
  %19 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %19, 3200
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: @hoo

; CHECK: for.body:
; CHECK: %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %arrayidx = getelementptr inbounds i32* %ip, i64 %indvar
; CHECK: %0 = load i32* %arrayidx, align 4
; CHECK: %idxprom1 = sext i32 %0 to i64
; CHECK: %arrayidx2 = getelementptr inbounds float* %b, i64 %idxprom1
; CHECK: %1 = load float* %arrayidx2, align 4
; CHECK: %mul = fmul float %1, %alpha
; CHECK: %arrayidx4 = getelementptr inbounds float* %a, i64 %indvar
; CHECK: %2 = load float* %arrayidx4, align 4
; CHECK: %add = fadd float %2, %mul
; CHECK: store float %add, float* %arrayidx4, align 4
; CHECK: %indvar.next = add i64 %indvar, 1
; CHECK: %exitcond = icmp eq i64 %indvar, 3199
; CHECK: br i1 %exitcond, label %for.end, label %for.body

; CHECK: ret

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }

