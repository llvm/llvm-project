; RUN: opt < %s -analyze -basicaa -da | FileCheck %s

; ModuleID = 'SymbolicSIV.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"


;;  for (long int i = 0; i < n; i++)
;;    A[2*i + n] = ...
;;    ... = A[3*i + 3*n];

define void @symbolicsiv0(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %entry ]
  %conv = trunc i64 %i.03 to i32
  %mul = shl nsw i64 %i.03, 1
  %add = add i64 %mul, %n
  %arrayidx = getelementptr inbounds i32* %A, i64 %add
  store i32 %conv, i32* %arrayidx, align 4
  %mul14 = add i64 %i.03, %n
  %add3 = mul i64 %mul14, 3
  %arrayidx4 = getelementptr inbounds i32* %A, i64 %add3
  %0 = load i32* %arrayidx4, align 4
; CHECK: da analyze - none!
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add nsw i64 %i.03, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;;  for (long int i = 0; i < n; i++)
;;    A[2*i + 5*n] = ...
;;    ... = A[3*i + 2*n];

define void @symbolicsiv1(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %entry ]
  %conv = trunc i64 %i.03 to i32
  %mul = shl nsw i64 %i.03, 1
  %mul1 = mul i64 %n, 5
  %add = add i64 %mul, %mul1
  %arrayidx = getelementptr inbounds i32* %A, i64 %add
  store i32 %conv, i32* %arrayidx, align 4
  %mul2 = mul nsw i64 %i.03, 3
  %mul3 = shl i64 %n, 1
  %add4 = add i64 %mul2, %mul3
  %arrayidx5 = getelementptr inbounds i32* %A, i64 %add4
  %0 = load i32* %arrayidx5, align 4
; CHECK: da analyze - none!
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add nsw i64 %i.03, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;;  for (long int i = 0; i < n; i++)
;;    A[2*i - n] = ...
;;    ... = A[-i + 2*n];

define void @symbolicsiv2(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %entry ]
  %conv = trunc i64 %i.03 to i32
  %mul = shl nsw i64 %i.03, 1
  %sub = sub i64 %mul, %n
  %arrayidx = getelementptr inbounds i32* %A, i64 %sub
  store i32 %conv, i32* %arrayidx, align 4
  %mul2 = shl i64 %n, 1
  %add = sub i64 %mul2, %i.03
  %arrayidx3 = getelementptr inbounds i32* %A, i64 %add
  %0 = load i32* %arrayidx3, align 4
; CHECK: da analyze - none!
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add nsw i64 %i.03, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;;  for (long int i = 0; i < n; i++)
;;    A[-2*i + n + 1] = ...
;;    ... = A[i - 2*n];

define void @symbolicsiv3(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %entry ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, -2
  %add = add i64 %mul, %n
  %add1 = add i64 %add, 1
  %arrayidx = getelementptr inbounds i32* %A, i64 %add1
  store i32 %conv, i32* %arrayidx, align 4
  %mul2 = shl i64 %n, 1
  %sub = sub i64 %i.03, %mul2
  %arrayidx3 = getelementptr inbounds i32* %A, i64 %sub
  %0 = load i32* %arrayidx3, align 4
; CHECK: da analyze - none!
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add nsw i64 %i.03, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;;  for (long int i = 0; i < n; i++)
;;    A[-2*i + 3*n] = ...
;;    ... = A[-i + n];

define void @symbolicsiv4(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %entry ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, -2
  %mul1 = mul i64 %n, 3
  %add = add i64 %mul, %mul1
  %arrayidx = getelementptr inbounds i32* %A, i64 %add
  store i32 %conv, i32* %arrayidx, align 4
  %add2 = sub i64 %n, %i.03
  %arrayidx3 = getelementptr inbounds i32* %A, i64 %add2
  %0 = load i32* %arrayidx3, align 4
; CHECK: da analyze - none!
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add nsw i64 %i.03, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;;  for (long int i = 0; i < n; i++)
;;    A[-2*i - 2*n] = ...
;;    ... = A[-i - n];

define void @symbolicsiv5(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %entry ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, -2
  %mul1 = shl i64 %n, 1
  %sub = sub i64 %mul, %mul1
  %arrayidx = getelementptr inbounds i32* %A, i64 %sub
  store i32 %conv, i32* %arrayidx, align 4
  %sub2 = sub nsw i64 0, %i.03
  %sub3 = sub i64 %sub2, %n
  %arrayidx4 = getelementptr inbounds i32* %A, i64 %sub3
  %0 = load i32* %arrayidx4, align 4
; CHECK: da analyze - none!
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add nsw i64 %i.03, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;; why doesn't SCEV package understand that n >= 0?
;;void weaktest(int *A, int *B, long unsigned n)
;;  for (long unsigned i = 0; i < n; i++)
;;    A[i + n + 1] = ...
;;    ... = A[-i];

define void @weaktest(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %entry ]
  %conv = trunc i64 %i.03 to i32
  %add = add i64 %i.03, %n
  %add1 = add i64 %add, 1
  %arrayidx = getelementptr inbounds i32* %A, i64 %add1
  store i32 %conv, i32* %arrayidx, align 4
  %sub = sub i64 0, %i.03
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %sub
  %0 = load i32* %arrayidx2, align 4
; CHECK: da analyze - flow [*|<] splitable!
; CHECK: da analyze - split level = 1, iteration = ((0 smax (-1 + (-1 * %n))) /u 2)!
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add i64 %i.03, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


;;  void symbolicsiv6(int *A, int *B, long unsigned n, long unsigned N, long unsigned M) {
;;    for (long int i = 0; i < n; i++) {
;;      A[4*N*i + M] = i;
;;      *B++ = A[4*N*i + 3*M + 1];

define void @symbolicsiv6(i32* %A, i32* %B, i64 %n, i64 %N, i64 %M) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %for.body.preheader ]
  %conv = trunc i64 %i.03 to i32
  %mul = shl i64 %N, 2
  %mul1 = mul i64 %mul, %i.03
  %add = add i64 %mul1, %M
  %arrayidx = getelementptr inbounds i32* %A, i64 %add
  store i32 %conv, i32* %arrayidx, align 4
  %mul2 = shl i64 %N, 2
  %mul3 = mul i64 %mul2, %i.03
  %mul4 = mul i64 %M, 3
  %add5 = add i64 %mul3, %mul4
  %add6 = add i64 %add5, 1
  %arrayidx7 = getelementptr inbounds i32* %A, i64 %add6
  %0 = load i32* %arrayidx7, align 4
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
; CHECK: da analyze - none!
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add nsw i64 %i.03, 1
  %exitcond = icmp ne i64 %inc, %n
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}


;;  void symbolicsiv7(int *A, int *B, long unsigned n, long unsigned N, long unsigned M) {
;;    for (long int i = 0; i < n; i++) {
;;      A[2*N*i + M] = i;
;;      *B++ = A[2*N*i - 3*M + 2];

define void @symbolicsiv7(i32* %A, i32* %B, i64 %n, i64 %N, i64 %M) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %for.body.preheader ]
  %conv = trunc i64 %i.03 to i32
  %mul = shl i64 %N, 1
  %mul1 = mul i64 %mul, %i.03
  %add = add i64 %mul1, %M
  %arrayidx = getelementptr inbounds i32* %A, i64 %add
  store i32 %conv, i32* %arrayidx, align 4
  %mul2 = shl i64 %N, 1
  %mul3 = mul i64 %mul2, %i.03
  %0 = mul i64 %M, -3
  %sub = add i64 %mul3, %0
  %add5 = add i64 %sub, 2
  %arrayidx6 = getelementptr inbounds i32* %A, i64 %add5
  %1 = load i32* %arrayidx6, align 4
  %incdec.ptr = getelementptr inbounds i32* %B.addr.02, i64 1
; CHECK: da analyze - flow [<>]!
  store i32 %1, i32* %B.addr.02, align 4
  %inc = add nsw i64 %i.03, 1
  %exitcond = icmp ne i64 %inc, %n
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
