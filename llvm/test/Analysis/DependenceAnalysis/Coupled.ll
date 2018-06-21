; RUN: opt < %s -analyze -basicaa -da | FileCheck %s

; ModuleID = 'Coupled.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"


;;  for (long int i = 0; i < 50; i++) {
;;    A[i][i] = i;
;;    *B++ = A[i + 10][i + 9];

define void @couple0([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %arrayidx1 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  store i32 %conv, i32* %arrayidx1, align 4
  %add = add nsw i64 %i.02, 9
  %add2 = add nsw i64 %i.02, 10
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %add2, i64 %add
  %0 = load i32, i32* %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i < 50; i++) {
;;    A[i][i] = i;
;;    *B++ = A[i + 9][i + 9];

define void @couple1([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - consistent flow [-9]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %arrayidx1 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  store i32 %conv, i32* %arrayidx1, align 4
  %add = add nsw i64 %i.02, 9
  %add2 = add nsw i64 %i.02, 9
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %add2, i64 %add
  %0 = load i32, i32* %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i < 50; i++) {
;;    A[3*i - 6][3*i - 6] = i;
;;    *B++ = A[i][i];

define void @couple2([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [*|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul nsw i64 %i.02, 3
  %sub = add nsw i64 %mul, -6
  %mul1 = mul nsw i64 %i.02, 3
  %sub2 = add nsw i64 %mul1, -6
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %sub2, i64 %sub
  store i32 %conv, i32* %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx5, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i < 50; i++) {
;;    A[3*i - 6][3*i - 5] = i;
;;    *B++ = A[i][i];

define void @couple3([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul nsw i64 %i.02, 3
  %sub = add nsw i64 %mul, -5
  %mul1 = mul nsw i64 %i.02, 3
  %sub2 = add nsw i64 %mul1, -6
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %sub2, i64 %sub
  store i32 %conv, i32* %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx5, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i < 50; i++) {
;;    A[3*i - 6][3*i - n] = i;
;;    *B++ = A[i][i];

define void @couple4([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [*|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul nsw i64 %i.02, 3
  %conv1 = sext i32 %n to i64
  %sub = sub nsw i64 %mul, %conv1
  %mul2 = mul nsw i64 %i.02, 3
  %sub3 = add nsw i64 %mul2, -6
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %sub3, i64 %sub
  store i32 %conv, i32* %arrayidx4, align 4
  %arrayidx6 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx6, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i < 50; i++) {
;;    A[3*i - n + 1][3*i - n] = i;
;;    *B++ = A[i][i];

define void @couple5([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul nsw i64 %i.02, 3
  %conv1 = sext i32 %n to i64
  %sub = sub nsw i64 %mul, %conv1
  %mul2 = mul nsw i64 %i.02, 3
  %conv3 = sext i32 %n to i64
  %sub4 = sub nsw i64 %mul2, %conv3
  %add = add nsw i64 %sub4, 1
  %arrayidx5 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %add, i64 %sub
  store i32 %conv, i32* %arrayidx5, align 4
  %arrayidx7 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx7, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i < 50; i++) {
;;    A[i][3*i - 6] = i;
;;    *B++ = A[i][i];

define void @couple6([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [=|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul nsw i64 %i.02, 3
  %sub = add nsw i64 %mul, -6
  %arrayidx1 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %sub
  store i32 %conv, i32* %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx3, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i < 50; i++) {
;;    A[i][3*i - 5] = i;
;;    *B++ = A[i][i];

define void @couple7([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul nsw i64 %i.02, 3
  %sub = add nsw i64 %mul, -5
  %arrayidx1 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %sub
  store i32 %conv, i32* %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx3, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 50
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i <= 15; i++) {
;;    A[3*i - 18][3 - i] = i;
;;    *B++ = A[i][i];

define void @couple8([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %sub = sub nsw i64 3, %i.02
  %mul = mul nsw i64 %i.02, 3
  %sub1 = add nsw i64 %mul, -18
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %sub1, i64 %sub
  store i32 %conv, i32* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 16
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i <= 15; i++) {
;;    A[3*i - 18][2 - i] = i;
;;    *B++ = A[i][i];

define void @couple9([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %sub = sub nsw i64 2, %i.02
  %mul = mul nsw i64 %i.02, 3
  %sub1 = add nsw i64 %mul, -18
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %sub1, i64 %sub
  store i32 %conv, i32* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 16
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i <= 15; i++) {
;;    A[3*i - 18][6 - i] = i;
;;    *B++ = A[i][i];

define void @couple10([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [>] splitable!
; CHECK: da analyze - split level = 1, iteration = 3!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %sub = sub nsw i64 6, %i.02
  %mul = mul nsw i64 %i.02, 3
  %sub1 = add nsw i64 %mul, -18
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %sub1, i64 %sub
  store i32 %conv, i32* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 16
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i <= 15; i++) {
;;    A[3*i - 18][18 - i] = i;
;;    *B++ = A[i][i];

define void @couple11([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [=|<] splitable!
; CHECK: da analyze - split level = 1, iteration = 9!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %sub = sub nsw i64 18, %i.02
  %mul = mul nsw i64 %i.02, 3
  %sub1 = add nsw i64 %mul, -18
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %sub1, i64 %sub
  store i32 %conv, i32* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 16
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i <= 12; i++) {
;;    A[3*i - 18][22 - i] = i;
;;    *B++ = A[i][i];

define void @couple12([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [<] splitable!
; CHECK: da analyze - split level = 1, iteration = 11!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %sub = sub nsw i64 22, %i.02
  %mul = mul nsw i64 %i.02, 3
  %sub1 = add nsw i64 %mul, -18
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %sub1, i64 %sub
  store i32 %conv, i32* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 13
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i < 12; i++) {
;;    A[3*i - 18][22 - i] = i;
;;    *B++ = A[i][i];

define void @couple13([100 x i32]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %sub = sub nsw i64 22, %i.02
  %mul = mul nsw i64 %i.02, 3
  %sub1 = add nsw i64 %mul, -18
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %sub1, i64 %sub
  store i32 %conv, i32* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 12
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

;;  for (long int i = 0; i < 100; i++) {
;;    A[3*i - 18][18 - i][i] = i;
;;    *B++ = A[i][i][i];

define void @couple14([100 x [100 x i32]]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [=|<] splitable!
; CHECK: da analyze - split level = 1, iteration = 9!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %sub = sub nsw i64 18, %i.02
  %mul = mul nsw i64 %i.02, 3
  %sub1 = add nsw i64 %mul, -18
  %arrayidx3 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %A, i64 %sub1, i64 %sub, i64 %i.02
  store i32 %conv, i32* %arrayidx3, align 4
  %arrayidx6 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %A, i64 %i.02, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx6, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 100
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long int i = 0; i < 100; i++) {
;;    A[3*i - 18][22 - i][i] = i;
;;    *B++ = A[i][i][i];

define void @couple15([100 x [100 x i32]]* %A, i32* %B, i32 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %sub = sub nsw i64 22, %i.02
  %mul = mul nsw i64 %i.02, 3
  %sub1 = add nsw i64 %mul, -18
  %arrayidx3 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %A, i64 %sub1, i64 %sub, i64 %i.02
  store i32 %conv, i32* %arrayidx3, align 4
  %arrayidx6 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %A, i64 %i.02, i64 %i.02, i64 %i.02
  %0 = load i32, i32* %arrayidx6, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add nsw i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 100
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}
