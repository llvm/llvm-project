; RUN: opt < %s -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -S | FileCheck %s
; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s -check-prefix=WIDTH
; RUN: opt -S -passes=loop-vectorize -force-vector-width=4 < %s | FileCheck %s -check-prefix=RIGHTVF
; RUN: opt -S -passes=loop-vectorize -force-vector-width=8 < %s | FileCheck %s -check-prefix=WRONGVF

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Vectorization with dependence checks.

; No plausible dependence - can be vectorized.
;  for (i = 0; i < 1024; ++i)
;    A[i] = A[i + 1] + 1;

; CHECK-LABEL: @f1_vec(
; CHECK: <2 x i32>

define void @f1_vec(ptr %A) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add i32 %indvars.iv, 1
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %indvars.iv.next
  %0 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %0, 1
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i32 %indvars.iv
  store i32 %add1, ptr %arrayidx3, align 4
  %exitcond = icmp ne i32 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; Plausible dependence of distance 1 - can't be vectorized.
;  for (i = 0; i < 1024; ++i)
;    A[i+1] = A[i] + 1;

; CHECK-LABEL: @f2_novec(
; CHECK-NOT: <2 x i32>

define void @f2_novec(ptr %A) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, 1
  %indvars.iv.next = add i32 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i32 %indvars.iv.next
  store i32 %add, ptr %arrayidx3, align 4
  %exitcond = icmp ne i32 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; Plausible dependence of distance 2 - can be vectorized with a width of 2.
;  for (i = 0; i < 1024; ++i)
;    A[i+2] = A[i] + 1;

; CHECK-LABEL: @f3_vec_len(
; CHECK: <2 x i32>

; WIDTH: f3_vec_len
; WIDTH-NOT: <4 x i32>

define void @f3_vec_len(ptr %A) {
entry:
  br label %for.body

for.body:
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %idxprom = sext i32 %i.01 to i64
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, 1
  %add1 = add nsw i32 %i.01, 2
  %idxprom2 = sext i32 %add1 to i64
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %idxprom2
  store i32 %add, ptr %arrayidx3, align 4
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 1024
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; Plausible dependence of distance 1 - cannot be vectorized (without reordering
; accesses).
;   for (i = 0; i < 1024; ++i) {
;     B[i] = A[i];
;     A[i] = B[i + 1];
;   }

; CHECK-LABEL: @f5(
; CHECK-NOT: <2 x i32>

define void @f5(ptr  %A, ptr %B) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  store i32 %0, ptr %arrayidx2, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %arrayidx4 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv.next
  %1 = load i32, ptr %arrayidx4, align 4
  store i32 %1, ptr %arrayidx, align 4
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; Dependence through a phi node - must not vectorize.
;   for (i = 0; i < 1024; ++i) {
;     a[i+1] = tmp;
;     tmp = a[i];
;   }

; CHECK-LABEL: @f6
; CHECK-NOT: <2 x i32>

define i32 @f6(ptr %a, i32 %tmp) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp.addr.08 = phi i32 [ %tmp, %entry ], [ %0, %for.body ]
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv.next
  store i32 %tmp.addr.08, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx3, align 4
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret i32 undef
}

; Don't vectorize true loop carried dependencies that are not a multiple of the
; vector width.
; Example:
;   for (int i = ...; ++i) {
;     a[i] = a[i-3] + ...;
; It is a bad idea to vectorize this loop because store-load forwarding will not
; happen.
;

; CHECK-LABEL: @nostoreloadforward(
; CHECK-NOT: <2 x i32>

define void @nostoreloadforward(ptr %A) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 16, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = add nsw i64 %indvars.iv, -3
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %0
  %1 = load i32, ptr %arrayidx, align 4
  %2 = add nsw i64 %indvars.iv, 4
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %2
  %3 = load i32, ptr %arrayidx2, align 4
  %add3 = add nsw i32 %3, %1
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %add3, ptr %arrayidx5, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; Example:
;   for (int i = ...; ++i) {
;     a[i] = b[i];
;     c[i] = a[i-3] + ...;
; It is a bad idea to vectorize this loop because store-load forwarding will not
; happen.
;

; CHECK-LABEL: @nostoreloadforward2(
; CHECK-NOT: <2 x i32>

define void @nostoreloadforward2(ptr noalias %A, ptr noalias %B, ptr noalias %C) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 16, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %0, ptr %arrayidx2, align 4
  %1 = add nsw i64 %indvars.iv, -3
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 %1
  %2 = load i32, ptr %arrayidx4, align 4
  %arrayidx6 = getelementptr inbounds i32, ptr %C, i64 %indvars.iv
  store i32 %2, ptr %arrayidx6, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}


;Check the new calculation of the maximum safe distance in bits which can be vectorized.
;The previous behavior did not take account that the stride was 2.
;Therefore the maxVF was computed as 8 instead of 4, as the dependence distance here is 6 iterations, given by |N-(N-12)|/2.

;#define M 32
;#define N 2 * M
;unsigned int a [N];
;void pr34283(){
;	unsigned int j=0;
;   for (j = 0; j < M - 6; ++j)
;    {
;        a[N - 2 * j] = 69;
;        a[N - 12 - 2 * j] = 7;
;    }
;
;}

; RIGHTVF-LABEL: @pr34283
; RIGHTVF: <4 x i64>

; WRONGVF-LABLE: @pr34283
; WRONGVF-NOT: <8 x i64>

@a = common local_unnamed_addr global [64 x i32] zeroinitializer, align 16

; Function Attrs: norecurse nounwind uwtable
define void @pr34283() local_unnamed_addr {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = shl i64 %indvars.iv, 1
  %1 = sub nuw nsw i64 64, %0
  %arrayidx = getelementptr inbounds [64 x i32], ptr @a, i64 0, i64 %1
  store i32 69, ptr %arrayidx, align 8
  %2 = sub nuw nsw i64 52, %0
  %arrayidx4 = getelementptr inbounds [64 x i32], ptr @a, i64 0, i64 %2
  store i32 7, ptr %arrayidx4, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 26
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

