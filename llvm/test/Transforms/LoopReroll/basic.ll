; RUN: opt < %s -loop-reroll -S | FileCheck %s
; RUN: opt < %s -passes=loop-reroll -S | FileCheck %s
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
  %exitcond = icmp sge i32 %add3, 500
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: @bar

; CHECK: for.body:
; CHECK: %indvar = phi i32 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %call = tail call i32 @foo(i32 %indvar) #1
; CHECK: %indvar.next = add i32 %indvar, 1
; CHECK: %exitcond1 = icmp eq i32 %indvar, 500
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
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %indvars.iv
  store i32 %call, i32* %arrayidx, align 4
  %call1 = tail call i32 @foo(i32 0) #1
  %0 = add nsw i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds i32, i32* %x, i64 %0
  store i32 %call1, i32* %arrayidx3, align 4
  %call4 = tail call i32 @foo(i32 0) #1
  %1 = add nsw i64 %indvars.iv, 2
  %arrayidx7 = getelementptr inbounds i32, i32* %x, i64 %1
  store i32 %call4, i32* %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 3
  %2 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %2, 1500
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: @hi1

; CHECK: for.body:
; CHECK: %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %0 = trunc i64 %indvar to i32
; CHECK: %call = tail call i32 @foo(i32 0) #1
; CHECK: %arrayidx = getelementptr inbounds i32, i32* %x, i64 %indvar
; CHECK: store i32 %call, i32* %arrayidx, align 4
; CHECK: %indvar.next = add i64 %indvar, 1
; CHECK: %exitcond = icmp eq i32 %0, 1499
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
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %0
  store i32 %call, i32* %arrayidx, align 4
  %call1 = tail call i32 @foo(i32 0) #1
  %1 = add nsw i64 %0, 1
  %arrayidx4 = getelementptr inbounds i32, i32* %x, i64 %1
  store i32 %call1, i32* %arrayidx4, align 4
  %call5 = tail call i32 @foo(i32 0) #1
  %2 = add nsw i64 %0, 2
  %arrayidx9 = getelementptr inbounds i32, i32* %x, i64 %2
  store i32 %call5, i32* %arrayidx9, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 500
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: @hi2

; CHECK: for.body:
; CHECK: %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK: %call = tail call i32 @foo(i32 0) #1
; CHECK: %arrayidx = getelementptr inbounds i32, i32* %x, i64 %indvars.iv
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
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %mul = fmul float %0, %alpha
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %1 = load float, float* %arrayidx2, align 4
  %add = fadd float %1, %mul
  store float %add, float* %arrayidx2, align 4
  %2 = add nsw i64 %indvars.iv, 1
  %arrayidx5 = getelementptr inbounds float, float* %b, i64 %2
  %3 = load float, float* %arrayidx5, align 4
  %mul6 = fmul float %3, %alpha
  %arrayidx9 = getelementptr inbounds float, float* %a, i64 %2
  %4 = load float, float* %arrayidx9, align 4
  %add10 = fadd float %4, %mul6
  store float %add10, float* %arrayidx9, align 4
  %5 = add nsw i64 %indvars.iv, 2
  %arrayidx13 = getelementptr inbounds float, float* %b, i64 %5
  %6 = load float, float* %arrayidx13, align 4
  %mul14 = fmul float %6, %alpha
  %arrayidx17 = getelementptr inbounds float, float* %a, i64 %5
  %7 = load float, float* %arrayidx17, align 4
  %add18 = fadd float %7, %mul14
  store float %add18, float* %arrayidx17, align 4
  %8 = add nsw i64 %indvars.iv, 3
  %arrayidx21 = getelementptr inbounds float, float* %b, i64 %8
  %9 = load float, float* %arrayidx21, align 4
  %mul22 = fmul float %9, %alpha
  %arrayidx25 = getelementptr inbounds float, float* %a, i64 %8
  %10 = load float, float* %arrayidx25, align 4
  %add26 = fadd float %10, %mul22
  store float %add26, float* %arrayidx25, align 4
  %11 = add nsw i64 %indvars.iv, 4
  %arrayidx29 = getelementptr inbounds float, float* %b, i64 %11
  %12 = load float, float* %arrayidx29, align 4
  %mul30 = fmul float %12, %alpha
  %arrayidx33 = getelementptr inbounds float, float* %a, i64 %11
  %13 = load float, float* %arrayidx33, align 4
  %add34 = fadd float %13, %mul30
  store float %add34, float* %arrayidx33, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 5
  %14 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %14, 3200
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: @goo

; CHECK: for.body:
; CHECK: %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %0 = trunc i64 %indvar to i32
; CHECK: %arrayidx = getelementptr inbounds float, float* %b, i64 %indvar
; CHECK: %1 = load float, float* %arrayidx, align 4
; CHECK: %mul = fmul float %1, %alpha
; CHECK: %arrayidx2 = getelementptr inbounds float, float* %a, i64 %indvar
; CHECK: %2 = load float, float* %arrayidx2, align 4
; CHECK: %add = fadd float %2, %mul
; CHECK: store float %add, float* %arrayidx2, align 4
; CHECK: %indvar.next = add i64 %indvar, 1
; CHECK: %exitcond = icmp eq i32 %0, 3199
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
  %arrayidx = getelementptr inbounds i32, i32* %ip, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %idxprom1 = sext i32 %0 to i64
  %arrayidx2 = getelementptr inbounds float, float* %b, i64 %idxprom1
  %1 = load float, float* %arrayidx2, align 4
  %mul = fmul float %1, %alpha
  %arrayidx4 = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %2 = load float, float* %arrayidx4, align 4
  %add = fadd float %2, %mul
  store float %add, float* %arrayidx4, align 4
  %3 = add nsw i64 %indvars.iv, 1
  %arrayidx7 = getelementptr inbounds i32, i32* %ip, i64 %3
  %4 = load i32, i32* %arrayidx7, align 4
  %idxprom8 = sext i32 %4 to i64
  %arrayidx9 = getelementptr inbounds float, float* %b, i64 %idxprom8
  %5 = load float, float* %arrayidx9, align 4
  %mul10 = fmul float %5, %alpha
  %arrayidx13 = getelementptr inbounds float, float* %a, i64 %3
  %6 = load float, float* %arrayidx13, align 4
  %add14 = fadd float %6, %mul10
  store float %add14, float* %arrayidx13, align 4
  %7 = add nsw i64 %indvars.iv, 2
  %arrayidx17 = getelementptr inbounds i32, i32* %ip, i64 %7
  %8 = load i32, i32* %arrayidx17, align 4
  %idxprom18 = sext i32 %8 to i64
  %arrayidx19 = getelementptr inbounds float, float* %b, i64 %idxprom18
  %9 = load float, float* %arrayidx19, align 4
  %mul20 = fmul float %9, %alpha
  %arrayidx23 = getelementptr inbounds float, float* %a, i64 %7
  %10 = load float, float* %arrayidx23, align 4
  %add24 = fadd float %10, %mul20
  store float %add24, float* %arrayidx23, align 4
  %11 = add nsw i64 %indvars.iv, 3
  %arrayidx27 = getelementptr inbounds i32, i32* %ip, i64 %11
  %12 = load i32, i32* %arrayidx27, align 4
  %idxprom28 = sext i32 %12 to i64
  %arrayidx29 = getelementptr inbounds float, float* %b, i64 %idxprom28
  %13 = load float, float* %arrayidx29, align 4
  %mul30 = fmul float %13, %alpha
  %arrayidx33 = getelementptr inbounds float, float* %a, i64 %11
  %14 = load float, float* %arrayidx33, align 4
  %add34 = fadd float %14, %mul30
  store float %add34, float* %arrayidx33, align 4
  %15 = add nsw i64 %indvars.iv, 4
  %arrayidx37 = getelementptr inbounds i32, i32* %ip, i64 %15
  %16 = load i32, i32* %arrayidx37, align 4
  %idxprom38 = sext i32 %16 to i64
  %arrayidx39 = getelementptr inbounds float, float* %b, i64 %idxprom38
  %17 = load float, float* %arrayidx39, align 4
  %mul40 = fmul float %17, %alpha
  %arrayidx43 = getelementptr inbounds float, float* %a, i64 %15
  %18 = load float, float* %arrayidx43, align 4
  %add44 = fadd float %18, %mul40
  store float %add44, float* %arrayidx43, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 5
  %19 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %19, 3200
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: @hoo

; CHECK: for.body:
; CHECK: %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %0 = trunc i64 %indvar to i32
; CHECK: %arrayidx = getelementptr inbounds i32, i32* %ip, i64 %indvar
; CHECK: %1 = load i32, i32* %arrayidx, align 4
; CHECK: %idxprom1 = sext i32 %1 to i64
; CHECK: %arrayidx2 = getelementptr inbounds float, float* %b, i64 %idxprom1
; CHECK: %2 = load float, float* %arrayidx2, align 4
; CHECK: %mul = fmul float %2, %alpha
; CHECK: %arrayidx4 = getelementptr inbounds float, float* %a, i64 %indvar
; CHECK: %3 = load float, float* %arrayidx4, align 4
; CHECK: %add = fadd float %3, %mul
; CHECK: store float %add, float* %arrayidx4, align 4
; CHECK: %indvar.next = add i64 %indvar, 1
; CHECK: %exitcond = icmp eq i32 %0, 3199
; CHECK: br i1 %exitcond, label %for.end, label %for.body

; CHECK: ret

for.end:                                          ; preds = %for.body
  ret void
}

; void multi1(int *x) {
;   y = foo(0)
;   for (int i = 0; i < 500; ++i) {
;     x[3*i] = y;
;     x[3*i+1] = y;
;     x[3*i+2] = y;
;     x[3*i+6] = y;
;     x[3*i+7] = y;
;     x[3*i+8] = y;
;   }
; }

; Function Attrs: nounwind uwtable
define void @multi1(i32* nocapture %x) #0 {
entry:
  %call = tail call i32 @foo(i32 0) #1
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = mul nsw i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %0
  store i32 %call, i32* %arrayidx, align 4
  %1 = add nsw i64 %0, 1
  %arrayidx4 = getelementptr inbounds i32, i32* %x, i64 %1
  store i32 %call, i32* %arrayidx4, align 4
  %2 = add nsw i64 %0, 2
  %arrayidx9 = getelementptr inbounds i32, i32* %x, i64 %2
  store i32 %call, i32* %arrayidx9, align 4
  %3 = add nsw i64 %0, 6
  %arrayidx6 = getelementptr inbounds i32, i32* %x, i64 %3
  store i32 %call, i32* %arrayidx6, align 4
  %4 = add nsw i64 %0, 7
  %arrayidx7 = getelementptr inbounds i32, i32* %x, i64 %4
  store i32 %call, i32* %arrayidx7, align 4
  %5 = add nsw i64 %0, 8
  %arrayidx8 = getelementptr inbounds i32, i32* %x, i64 %5
  store i32 %call, i32* %arrayidx8, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 500
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: @multi1

; CHECK:for.body:
; CHECK:  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK:  %0 = add i64 %indvars.iv, 6
; CHECK:  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %indvars.iv
; CHECK:  store i32 %call, i32* %arrayidx, align 4
; CHECK:  %arrayidx6 = getelementptr inbounds i32, i32* %x, i64 %0
; CHECK:  store i32 %call, i32* %arrayidx6, align 4
; CHECK:  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:  %exitcond1 = icmp eq i64 %indvars.iv, 1499
; CHECK:  br i1 %exitcond1, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; void multi2(int *x) {
;   y = foo(0)
;   for (int i = 0; i < 500; ++i) {
;     x[3*i] = y;
;     x[3*i+1] = y;
;     x[3*i+2] = y;
;     x[3*(i+1)] = y;
;     x[3*(i+1)+1] = y;
;     x[3*(i+1)+2] = y;
;   }
; }

; Function Attrs: nounwind uwtable
define void @multi2(i32* nocapture %x) #0 {
entry:
  %call = tail call i32 @foo(i32 0) #1
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = mul nsw i64 %indvars.iv, 3
  %add = add nsw i64 %indvars.iv, 1
  %newmul = mul nsw i64 %add, 3
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %0
  store i32 %call, i32* %arrayidx, align 4
  %1 = add nsw i64 %0, 1
  %arrayidx4 = getelementptr inbounds i32, i32* %x, i64 %1
  store i32 %call, i32* %arrayidx4, align 4
  %2 = add nsw i64 %0, 2
  %arrayidx9 = getelementptr inbounds i32, i32* %x, i64 %2
  store i32 %call, i32* %arrayidx9, align 4
  %arrayidx6 = getelementptr inbounds i32, i32* %x, i64 %newmul
  store i32 %call, i32* %arrayidx6, align 4
  %3 = add nsw i64 %newmul, 1
  %arrayidx7 = getelementptr inbounds i32, i32* %x, i64 %3
  store i32 %call, i32* %arrayidx7, align 4
  %4 = add nsw i64 %newmul, 2
  %arrayidx8 = getelementptr inbounds i32, i32* %x, i64 %4
  store i32 %call, i32* %arrayidx8, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 500
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: @multi2

; CHECK:for.body:
; CHECK:  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK:  %0 = add i64 %indvars.iv, 3
; CHECK:  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %indvars.iv
; CHECK:  store i32 %call, i32* %arrayidx, align 4
; CHECK:  %arrayidx6 = getelementptr inbounds i32, i32* %x, i64 %0
; CHECK:  store i32 %call, i32* %arrayidx6, align 4
; CHECK:  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:  %exitcond1 = icmp eq i64 %indvars.iv, 1499
; CHECK:  br i1 %exitcond1, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; void multi3(int *x) {
;   y = foo(0)
;   for (int i = 0; i < 500; ++i) {
;     // Note: No zero index
;     x[3*i+3] = y;
;     x[3*i+4] = y;
;     x[3*i+5] = y;
;   }
; }

; Function Attrs: nounwind uwtable
define void @multi3(i32* nocapture %x) #0 {
entry:
  %call = tail call i32 @foo(i32 0) #1
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = mul nsw i64 %indvars.iv, 3
  %x0 = add nsw i64 %0, 3
  %add = add nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %x0
  store i32 %call, i32* %arrayidx, align 4
  %1 = add nsw i64 %0, 4
  %arrayidx4 = getelementptr inbounds i32, i32* %x, i64 %1
  store i32 %call, i32* %arrayidx4, align 4
  %2 = add nsw i64 %0, 5
  %arrayidx9 = getelementptr inbounds i32, i32* %x, i64 %2
  store i32 %call, i32* %arrayidx9, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 500
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: @multi3
; CHECK: for.body:
; CHECK:   %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK:   %0 = add i64 %indvars.iv, 3
; CHECK:   %arrayidx = getelementptr inbounds i32, i32* %x, i64 %0
; CHECK:   store i32 %call, i32* %arrayidx, align 4
; CHECK:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:   %exitcond1 = icmp eq i64 %indvars.iv, 1499
; CHECK:   br i1 %exitcond1, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; int foo(int a);
; void bar2(int *x, int y, int z) {
;   for (int i = 0; i < 500; i += 3) {
;     foo(i+y+i*z); // Slightly reordered instruction order
;     foo(i+1+y+(i+1)*z);
;     foo(i+2+y+(i+2)*z);
;   }
; }

; Function Attrs: nounwind uwtable
define void @bar2(i32* nocapture readnone %x, i32 %y, i32 %z) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %add3, %for.body ]

  %tmp1 = add i32 %i.08, %y
  %tmp2 = mul i32 %i.08, %z
  %tmp3 = add i32 %tmp2, %tmp1
  %call = tail call i32 @foo(i32 %tmp3) #1

  %add = add nsw i32 %i.08, 1
  %tmp2a = mul i32 %add, %z
  %tmp1a = add i32 %add, %y
  %tmp3a = add i32 %tmp2a, %tmp1a
  %calla = tail call i32 @foo(i32 %tmp3a) #1
  
  %add2 = add nsw i32 %i.08, 2
  %tmp2b = mul i32 %add2, %z
  %tmp1b = add i32 %add2, %y
  %tmp3b = add i32 %tmp2b, %tmp1b
  %callb = tail call i32 @foo(i32 %tmp3b) #1

  %add3 = add nsw i32 %i.08, 3

  %exitcond = icmp sge i32 %add3, 500
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: @bar2

; CHECK: for.body:
; CHECK: %indvar = phi i32 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %tmp1 = add i32 %indvar, %y
; CHECK: %tmp2 = mul i32 %indvar, %z
; CHECK: %tmp3 = add i32 %tmp2, %tmp1
; CHECK: %call = tail call i32 @foo(i32 %tmp3) #1
; CHECK: %indvar.next = add i32 %indvar, 1
; CHECK: %exitcond1 = icmp eq i32 %indvar, 500
; CHECK: br i1 %exitcond1, label %for.end, label %for.body

; CHECK: ret

for.end:                                          ; preds = %for.body
  ret void
}

%struct.s = type { i32, i32 }

; Function Attrs: nounwind uwtable
define void @gep1(%struct.s* nocapture %x) #0 {
entry:
  %call = tail call i32 @foo(i32 0) #1
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = mul nsw i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds %struct.s, %struct.s* %x, i64 %0, i32 0
  store i32 %call, i32* %arrayidx, align 4
  %1 = add nsw i64 %0, 1
  %arrayidx4 = getelementptr inbounds %struct.s, %struct.s* %x, i64 %1, i32 0
  store i32 %call, i32* %arrayidx4, align 4
  %2 = add nsw i64 %0, 2
  %arrayidx9 = getelementptr inbounds %struct.s, %struct.s* %x, i64 %2, i32 0
  store i32 %call, i32* %arrayidx9, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 500
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: @gep1
; This test is a crash test only.
; CHECK: ret
for.end:                                          ; preds = %for.body
  ret void
}

define void @gep-indexing(i32* nocapture %x) {
entry:
  %call = tail call i32 @foo(i32 0) #1
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = mul nsw i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %0
  store i32 %call, i32* %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %arrayidx, i64 1
  store i32 %call, i32* %arrayidx4, align 4
  %arrayidx9 = getelementptr inbounds i32, i32* %arrayidx, i64 2
  store i32 %call, i32* %arrayidx9, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 500
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: @gep-indexing
; CHECK:      for.body:
; CHECK-NEXT:   %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT:   %scevgep = getelementptr i32, i32* %x, i64 %indvars.iv
; CHECK-NEXT:   store i32 %call, i32* %scevgep, align 4
; CHECK-NEXT:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-NEXT:   %exitcond1 = icmp eq i64 %indvars.iv, 1499
; CHECK-NEXT:   br i1 %exitcond1, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}


define void @unordered_atomic_ops(i32* noalias %buf_0, i32* noalias %buf_1) {
; CHECK-LABEL: @unordered_atomic_ops(

; CHECK: for.body:
; CHECK-NEXT:   %indvar = phi i32 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %buf0_a = getelementptr i32, i32* %buf_0, i32 %indvar
; CHECK-NEXT:   %buf1_a = getelementptr i32, i32* %buf_1, i32 %indvar
; CHECK-NEXT:   %va = load atomic i32, i32* %buf0_a unordered, align 4
; CHECK-NEXT:   store atomic i32 %va, i32* %buf1_a unordered, align 4
; CHECK-NEXT:   %indvar.next = add i32 %indvar, 1
; CHECK-NEXT:   %exitcond = icmp eq i32 %indvar, 3199
; CHECK-NEXT:   br i1 %exitcond, label %for.end, label %for.body

entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add i32 %indvars.iv, 2
  %indvars.mid = add i32 %indvars.iv, 1
  %buf0_a = getelementptr i32, i32* %buf_0, i32 %indvars.iv
  %buf0_b = getelementptr i32, i32* %buf_0, i32 %indvars.mid
  %buf1_a = getelementptr i32, i32* %buf_1, i32 %indvars.iv
  %buf1_b = getelementptr i32, i32* %buf_1, i32 %indvars.mid
  %va = load atomic i32, i32* %buf0_a unordered, align 4
  %vb = load atomic i32, i32* %buf0_b unordered, align 4
  store atomic i32 %va, i32* %buf1_a unordered, align 4
  store atomic i32 %vb, i32* %buf1_b unordered, align 4
  %cmp = icmp slt i32 %indvars.iv.next, 3200
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

define void @unordered_atomic_ops_nomatch(i32* noalias %buf_0, i32* noalias %buf_1) {
; Negative test

; CHECK-LABEL: @unordered_atomic_ops_nomatch(
entry:
  br label %for.body

for.body:
; CHECK: for.body:
; CHECK:   %indvars.iv.next = add i32 %indvars.iv, 2
; CHECK:   %indvars.mid = add i32 %indvars.iv, 1
; CHECK:   %cmp = icmp slt i32 %indvars.iv.next, 3200
; CHECK:   br i1 %cmp, label %for.body, label %for.end

  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add i32 %indvars.iv, 2
  %indvars.mid = add i32 %indvars.iv, 1
  %buf0_a = getelementptr i32, i32* %buf_0, i32 %indvars.iv
  %buf0_b = getelementptr i32, i32* %buf_0, i32 %indvars.mid
  %buf1_a = getelementptr i32, i32* %buf_1, i32 %indvars.iv
  %buf1_b = getelementptr i32, i32* %buf_1, i32 %indvars.mid
  %va = load atomic i32, i32* %buf0_a unordered, align 4
  %vb = load atomic i32, i32* %buf0_b unordered, align 4
  store i32 %va, i32* %buf1_a, align 4  ;; Not atomic
  store atomic i32 %vb, i32* %buf1_b unordered, align 4
  %cmp = icmp slt i32 %indvars.iv.next, 3200
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

define void @ordered_atomic_ops(i32* noalias %buf_0, i32* noalias %buf_1) {
; Negative test

; CHECK-LABEL: @ordered_atomic_ops(
entry:
  br label %for.body

for.body:
; CHECK: for.body:
; CHECK:   %indvars.iv.next = add i32 %indvars.iv, 2
; CHECK:   %indvars.mid = add i32 %indvars.iv, 1
; CHECK:   %cmp = icmp slt i32 %indvars.iv.next, 3200
; CHECK:   br i1 %cmp, label %for.body, label %for.end

  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add i32 %indvars.iv, 2
  %indvars.mid = add i32 %indvars.iv, 1
  %buf0_a = getelementptr i32, i32* %buf_0, i32 %indvars.iv
  %buf0_b = getelementptr i32, i32* %buf_0, i32 %indvars.mid
  %buf1_a = getelementptr i32, i32* %buf_1, i32 %indvars.iv
  %buf1_b = getelementptr i32, i32* %buf_1, i32 %indvars.mid
  %va = load atomic i32, i32* %buf0_a acquire, align 4
  %vb = load atomic i32, i32* %buf0_b acquire, align 4
  store atomic i32 %va, i32* %buf1_a release, align 4
  store atomic i32 %vb, i32* %buf1_b release, align 4
  %cmp = icmp slt i32 %indvars.iv.next, 3200
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

define void @unordered_atomic_ops_with_fence(i32* noalias %buf_0, i32* noalias %buf_1) {
; CHECK-LABEL: @unordered_atomic_ops_with_fence(
entry:
  br label %for.body

for.body:
; CHECK: for.body:
; CHECK:  %va = load atomic i32, i32* %buf0_a unordered, align 4
; CHECK-NEXT:  %vb = load atomic i32, i32* %buf0_b unordered, align 4
; CHECK-NEXT:  fence seq_cst
; CHECK-NEXT:  store atomic i32 %va, i32* %buf1_a unordered, align 4
; CHECK-NEXT:  store atomic i32 %vb, i32* %buf1_b unordered, align 4

  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add i32 %indvars.iv, 2
  %indvars.mid = add i32 %indvars.iv, 1
  %buf0_a = getelementptr i32, i32* %buf_0, i32 %indvars.iv
  %buf0_b = getelementptr i32, i32* %buf_0, i32 %indvars.mid
  %buf1_a = getelementptr i32, i32* %buf_1, i32 %indvars.iv
  %buf1_b = getelementptr i32, i32* %buf_1, i32 %indvars.mid
  %va = load atomic i32, i32* %buf0_a unordered, align 4
  %vb = load atomic i32, i32* %buf0_b unordered, align 4
  fence seq_cst
  store atomic i32 %va, i32* %buf1_a unordered, align 4
  store atomic i32 %vb, i32* %buf1_b unordered, align 4
  %cmp = icmp slt i32 %indvars.iv.next, 3200
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

define void @pointer_bitcast_baseinst(i16* %arg, i8* %arg1, i64 %arg2) {
; CHECK-LABEL: @pointer_bitcast_baseinst(
; CHECK:       bb3:
; CHECK-NEXT:    %indvar = phi i64 [ %indvar.next, %bb3 ], [ 0, %bb ]
; CHECK-NEXT:    %4 = shl nuw i64 %indvar, 3
; CHECK-NEXT:    %5 = add i64 %4, 1
; CHECK-NEXT:    %tmp5 = shl nuw i64 %5, 1
; CHECK-NEXT:    %tmp6 = getelementptr i8, i8* %arg1, i64 %tmp5
; CHECK-NEXT:    %tmp7 = bitcast i8* %tmp6 to <8 x i16>*
; CHECK-NEXT:    %tmp8 = load <8 x i16>, <8 x i16>* %tmp7, align 2
; CHECK-NEXT:    %tmp13 = getelementptr i16, i16* %arg, i64 %5
; CHECK-NEXT:    %tmp14 = bitcast i16* %tmp13 to <8 x i16>*
; CHECK-NEXT:    store <8 x i16> %tmp8, <8 x i16>* %tmp14, align 2
; CHECK-NEXT:    %indvar.next = add i64 %indvar, 1
; CHECK-NEXT:    %exitcond = icmp eq i64 %indvar, %3
; CHECK-NEXT:    br i1 %exitcond, label %bb19, label %bb3
bb:
  br label %bb3

bb3:                                              ; preds = %bb3, %bb
  %tmp = phi i64 [ 1, %bb ], [ %tmp17, %bb3 ]
  %tmp4 = add nuw i64 %tmp, 8
  %tmp5 = shl nuw i64 %tmp, 1
  %tmp6 = getelementptr i8, i8* %arg1, i64 %tmp5
  %tmp7 = bitcast i8* %tmp6 to <8 x i16>*
  %tmp8 = load <8 x i16>, <8 x i16>* %tmp7, align 2
  %tmp9 = shl i64 %tmp4, 1
  %tmp10 = getelementptr i8, i8* %arg1, i64 %tmp9
  %tmp11 = bitcast i8* %tmp10 to <8 x i16>*
  %tmp12 = load <8 x i16>, <8 x i16>* %tmp11, align 2
  %tmp13 = getelementptr i16, i16* %arg, i64 %tmp
  %tmp14 = bitcast i16* %tmp13 to <8 x i16>*
  store <8 x i16> %tmp8, <8 x i16>* %tmp14, align 2
  %tmp15 = getelementptr i16, i16* %arg, i64 %tmp4
  %tmp16 = bitcast i16* %tmp15 to <8 x i16>*
  store <8 x i16> %tmp12, <8 x i16>* %tmp16, align 2
  %tmp17 = add nuw nsw i64 %tmp, 16
  %tmp18 = icmp eq i64 %tmp17, %arg2
  br i1 %tmp18, label %bb19, label %bb3

bb19:                                             ; preds = %bb3
  ret void
}

define void @bad_step(i32* nocapture readnone %x) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %call = tail call i32 @foo(i32 %i.08) #1
  %add = add nsw i32 %i.08, 2
  %call1 = tail call i32 @foo(i32 %add) #1
  %add2 = add nsw i32 %i.08, 3
  %call3 = tail call i32 @foo(i32 %add2) #1
  %add3 = add nsw i32 %i.08, 6
  %exitcond = icmp sge i32 %add3, 500
  br i1 %exitcond, label %for.end, label %for.body

; CHECK-LABEL: @bad_step
; CHECK: %add = add nsw i32 %i.08, 2
; CHECK: %add2 = add nsw i32 %i.08, 3
; CHECK: %add3 = add nsw i32 %i.08, 6

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }

