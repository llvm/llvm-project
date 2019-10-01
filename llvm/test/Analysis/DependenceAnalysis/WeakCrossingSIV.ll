; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s
; RUN: opt < %s -analyze -basicaa -da | FileCheck %s

; ModuleID = 'WeakCrossingSIV.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"


;;  for (long unsigned i = 0; i < n; i++) {
;;    A[1 + n*i] = i;
;;    *B++ = A[1 - n*i];

define void @weakcrossing0(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [0|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %for.body.preheader ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul i64 %i.03, %n
  %add = add i64 %mul, 1
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %add
  store i32 %conv, i32* %arrayidx, align 4
  %mul1 = mul i64 %i.03, %n
  %sub = sub i64 1, %mul1
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %sub
  %0 = load i32, i32* %arrayidx2, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add i64 %i.03, 1
  %exitcond = icmp ne i64 %inc, %n
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}


;;  for (long unsigned i = 0; i < n; i++) {
;;    A[n + i] = i;
;;    *B++ = A[1 + n - i];

define void @weakcrossing1(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body.preheader

; CHECK: da analyze - none!
; CHECK: da analyze - flow [<>] splitable!
; CHECK: da analyze - split level = 1, iteration = 0!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %for.body.preheader ]
  %conv = trunc i64 %i.03 to i32
  %add = add i64 %i.03, %n
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %add
  store i32 %conv, i32* %arrayidx, align 4
  %add1 = add i64 %n, 1
  %sub = sub i64 %add1, %i.03
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %sub
  %0 = load i32, i32* %arrayidx2, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.02, i64 1
  store i32 %0, i32* %B.addr.02, align 4
  %inc = add i64 %i.03, 1
  %exitcond = icmp ne i64 %inc, %n
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}


;;  for (long unsigned i = 0; i < 3; i++) {
;;    A[i] = i;
;;    *B++ = A[6 - i];

define void @weakcrossing2(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
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
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.02
  store i32 %conv, i32* %arrayidx, align 4
  %sub = sub i64 6, %i.02
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %sub
  %0 = load i32, i32* %arrayidx1, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 3
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i < 4; i++) {
;;    A[i] = i;
;;    *B++ = A[6 - i];

define void @weakcrossing3(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [0|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.02
  store i32 %conv, i32* %arrayidx, align 4
  %sub = sub i64 6, %i.02
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %sub
  %0 = load i32, i32* %arrayidx1, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 4
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i < 10; i++) {
;;    A[i] = i;
;;    *B++ = A[-6 - i];

define void @weakcrossing4(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
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
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.02
  store i32 %conv, i32* %arrayidx, align 4
  %sub = sub i64 -6, %i.02
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %sub
  %0 = load i32, i32* %arrayidx1, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i < n; i++) {
;;    A[3*i] = i;
;;    *B++ = A[5 - 3*i];

define void @weakcrossing5(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %cmp1 = icmp eq i64 %n, 0
  br i1 %cmp1, label %for.end, label %for.body.preheader

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %B.addr.02 = phi i32* [ %incdec.ptr, %for.body ], [ %B, %for.body.preheader ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul i64 %i.03, 3
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %0 = mul i64 %i.03, -3
  %sub = add i64 %0, 5
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %sub
  %1 = load i32, i32* %arrayidx2, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.02, i64 1
  store i32 %1, i32* %B.addr.02, align 4
  %inc = add i64 %i.03, 1
  %exitcond = icmp ne i64 %inc, %n
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}


;;  for (long unsigned i = 0; i < 4; i++) {
;;    A[i] = i;
;;    *B++ = A[5 - i];

define void @weakcrossing6(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [<>] splitable!
; CHECK: da analyze - split level = 1, iteration = 2!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.02
  store i32 %conv, i32* %arrayidx, align 4
  %sub = sub i64 5, %i.02
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %sub
  %0 = load i32, i32* %arrayidx1, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 4
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}
