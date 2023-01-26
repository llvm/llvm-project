; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s

; ModuleID = 'ExactRDIV.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"


;;  for (long int i = 0; i < 10; i++)
;;    A[4*i + 10] = i;
;;  for (long int j = 0; j < 10; j++)
;;    *B++ = A[2*j + 1];

define void @rdiv0(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %conv = trunc i64 %i.03 to i32
  %mul = shl nsw i64 %i.03, 2
  %add = add nsw i64 %mul, 10
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %add
  store i32 %conv, ptr %arrayidx, align 4
  %inc = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc, 10
  br i1 %exitcond5, label %for.body, label %for.body4.preheader

for.body4.preheader:                              ; preds = %for.body
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %j.02 = phi i64 [ %inc9, %for.body4 ], [ 0, %for.body4.preheader ]
  %B.addr.01 = phi ptr [ %incdec.ptr, %for.body4 ], [ %B, %for.body4.preheader ]
  %mul5 = shl nsw i64 %j.02, 1
  %add64 = or i64 %mul5, 1
  %arrayidx7 = getelementptr inbounds i32, ptr %A, i64 %add64
  %0 = load i32, ptr %arrayidx7, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.01, i64 1
  store i32 %0, ptr %B.addr.01, align 4
  %inc9 = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc9, 10
  br i1 %exitcond, label %for.body4, label %for.end10

for.end10:                                        ; preds = %for.body4
  ret void
}


;;  for (long int i = 0; i < 5; i++)
;;    A[11*i - 45] = i;
;;  for (long int j = 0; j < 10; j++)
;;    *B++ = A[j];

define void @rdiv1(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, 11
  %sub = add nsw i64 %mul, -45
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %sub
  store i32 %conv, ptr %arrayidx, align 4
  %inc = add nsw i64 %i.03, 1
  %exitcond4 = icmp ne i64 %inc, 5
  br i1 %exitcond4, label %for.body, label %for.body4.preheader

for.body4.preheader:                              ; preds = %for.body
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %j.02 = phi i64 [ %inc7, %for.body4 ], [ 0, %for.body4.preheader ]
  %B.addr.01 = phi ptr [ %incdec.ptr, %for.body4 ], [ %B, %for.body4.preheader ]
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %j.02
  %0 = load i32, ptr %arrayidx5, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.01, i64 1
  store i32 %0, ptr %B.addr.01, align 4
  %inc7 = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc7, 10
  br i1 %exitcond, label %for.body4, label %for.end8

for.end8:                                         ; preds = %for.body4
  ret void
}


;;  for (long int i = 0; i <= 5; i++)
;;    A[11*i - 45] = i;
;;  for (long int j = 0; j < 10; j++)
;;    *B++ = A[j];

define void @rdiv2(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, 11
  %sub = add nsw i64 %mul, -45
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %sub
  store i32 %conv, ptr %arrayidx, align 4
  %inc = add nsw i64 %i.03, 1
  %exitcond4 = icmp ne i64 %inc, 6
  br i1 %exitcond4, label %for.body, label %for.body4.preheader

for.body4.preheader:                              ; preds = %for.body
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %j.02 = phi i64 [ %inc7, %for.body4 ], [ 0, %for.body4.preheader ]
  %B.addr.01 = phi ptr [ %incdec.ptr, %for.body4 ], [ %B, %for.body4.preheader ]
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %j.02
  %0 = load i32, ptr %arrayidx5, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.01, i64 1
  store i32 %0, ptr %B.addr.01, align 4
  %inc7 = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc7, 10
  br i1 %exitcond, label %for.body4, label %for.end8

for.end8:                                         ; preds = %for.body4
  ret void
}


;;  for (long int i = 0; i < 5; i++)
;;    A[11*i - 45] = i;
;;  for (long int j = 0; j <= 10; j++)
;;    *B++ = A[j];

define void @rdiv3(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, 11
  %sub = add nsw i64 %mul, -45
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %sub
  store i32 %conv, ptr %arrayidx, align 4
  %inc = add nsw i64 %i.03, 1
  %exitcond4 = icmp ne i64 %inc, 5
  br i1 %exitcond4, label %for.body, label %for.body4.preheader

for.body4.preheader:                              ; preds = %for.body
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %j.02 = phi i64 [ %inc7, %for.body4 ], [ 0, %for.body4.preheader ]
  %B.addr.01 = phi ptr [ %incdec.ptr, %for.body4 ], [ %B, %for.body4.preheader ]
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %j.02
  %0 = load i32, ptr %arrayidx5, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.01, i64 1
  store i32 %0, ptr %B.addr.01, align 4
  %inc7 = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc7, 11
  br i1 %exitcond, label %for.body4, label %for.end8

for.end8:                                         ; preds = %for.body4
  ret void
}


;;  for (long int i = 0; i <= 5; i++)
;;    A[11*i - 45] = i;
;;  for (long int j = 0; j <= 10; j++)
;;    *B++ = A[j];

define void @rdiv4(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, 11
  %sub = add nsw i64 %mul, -45
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %sub
  store i32 %conv, ptr %arrayidx, align 4
  %inc = add nsw i64 %i.03, 1
  %exitcond4 = icmp ne i64 %inc, 6
  br i1 %exitcond4, label %for.body, label %for.body4.preheader

for.body4.preheader:                              ; preds = %for.body
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %j.02 = phi i64 [ %inc7, %for.body4 ], [ 0, %for.body4.preheader ]
  %B.addr.01 = phi ptr [ %incdec.ptr, %for.body4 ], [ %B, %for.body4.preheader ]
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %j.02
  %0 = load i32, ptr %arrayidx5, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.01, i64 1
  store i32 %0, ptr %B.addr.01, align 4
  %inc7 = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc7, 11
  br i1 %exitcond, label %for.body4, label %for.end8

for.end8:                                         ; preds = %for.body4
  ret void
}


;;  for (long int i = 0; i < 5; i++)
;;    A[-11*i + 45] = i;
;;  for (long int j = 0; j < 10; j++)
;;    *B++ = A[-j];

define void @rdiv5(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, -11
  %add = add nsw i64 %mul, 45
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %add
  store i32 %conv, ptr %arrayidx, align 4
  %inc = add nsw i64 %i.03, 1
  %exitcond4 = icmp ne i64 %inc, 5
  br i1 %exitcond4, label %for.body, label %for.body4.preheader

for.body4.preheader:                              ; preds = %for.body
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %j.02 = phi i64 [ %inc7, %for.body4 ], [ 0, %for.body4.preheader ]
  %B.addr.01 = phi ptr [ %incdec.ptr, %for.body4 ], [ %B, %for.body4.preheader ]
  %sub = sub nsw i64 0, %j.02
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %sub
  %0 = load i32, ptr %arrayidx5, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.01, i64 1
  store i32 %0, ptr %B.addr.01, align 4
  %inc7 = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc7, 10
  br i1 %exitcond, label %for.body4, label %for.end8

for.end8:                                         ; preds = %for.body4
  ret void
}


;;  for (long int i = 0; i <= 5; i++)
;;    A[-11*i + 45] = i;
;;  for (long int j = 0; j < 10; j++)
;;    *B++ = A[-j];

define void @rdiv6(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, -11
  %add = add nsw i64 %mul, 45
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %add
  store i32 %conv, ptr %arrayidx, align 4
  %inc = add nsw i64 %i.03, 1
  %exitcond4 = icmp ne i64 %inc, 6
  br i1 %exitcond4, label %for.body, label %for.body4.preheader

for.body4.preheader:                              ; preds = %for.body
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %j.02 = phi i64 [ %inc7, %for.body4 ], [ 0, %for.body4.preheader ]
  %B.addr.01 = phi ptr [ %incdec.ptr, %for.body4 ], [ %B, %for.body4.preheader ]
  %sub = sub nsw i64 0, %j.02
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %sub
  %0 = load i32, ptr %arrayidx5, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.01, i64 1
  store i32 %0, ptr %B.addr.01, align 4
  %inc7 = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc7, 10
  br i1 %exitcond, label %for.body4, label %for.end8

for.end8:                                         ; preds = %for.body4
  ret void
}


;;  for (long int i = 0; i < 5; i++)
;;    A[-11*i + 45] = i;
;;  for (long int j = 0; j <= 10; j++)
;;    *B++ = A[-j];

define void @rdiv7(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, -11
  %add = add nsw i64 %mul, 45
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %add
  store i32 %conv, ptr %arrayidx, align 4
  %inc = add nsw i64 %i.03, 1
  %exitcond4 = icmp ne i64 %inc, 5
  br i1 %exitcond4, label %for.body, label %for.body4.preheader

for.body4.preheader:                              ; preds = %for.body
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %j.02 = phi i64 [ %inc7, %for.body4 ], [ 0, %for.body4.preheader ]
  %B.addr.01 = phi ptr [ %incdec.ptr, %for.body4 ], [ %B, %for.body4.preheader ]
  %sub = sub nsw i64 0, %j.02
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %sub
  %0 = load i32, ptr %arrayidx5, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.01, i64 1
  store i32 %0, ptr %B.addr.01, align 4
  %inc7 = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc7, 11
  br i1 %exitcond, label %for.body4, label %for.end8

for.end8:                                         ; preds = %for.body4
  ret void
}


;;  for (long int i = 0; i <= 5; i++)
;;    A[-11*i + 45] = i;
;;  for (long int j = 0; j <= 10; j++)
;;    *B++ = A[-j];

define void @rdiv8(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK: da analyze - none!
; CHECK: da analyze - flow [|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, -11
  %add = add nsw i64 %mul, 45
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %add
  store i32 %conv, ptr %arrayidx, align 4
  %inc = add nsw i64 %i.03, 1
  %exitcond4 = icmp ne i64 %inc, 6
  br i1 %exitcond4, label %for.body, label %for.body4.preheader

for.body4.preheader:                              ; preds = %for.body
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %j.02 = phi i64 [ %inc7, %for.body4 ], [ 0, %for.body4.preheader ]
  %B.addr.01 = phi ptr [ %incdec.ptr, %for.body4 ], [ %B, %for.body4.preheader ]
  %sub = sub nsw i64 0, %j.02
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %sub
  %0 = load i32, ptr %arrayidx5, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.01, i64 1
  store i32 %0, ptr %B.addr.01, align 4
  %inc7 = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc7, 11
  br i1 %exitcond, label %for.body4, label %for.end8

for.end8:                                         ; preds = %for.body4
  ret void
}


;;  for (long int i = 0; i < 5; i++)
;;    for (long int j = 0; j < 10; j++) {
;;      A[11*i - j] = i;
;;      *B++ = A[45];

define void @rdiv9(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - consistent input [S S]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc5
  %B.addr.04 = phi ptr [ %B, %entry ], [ %scevgep, %for.inc5 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc6, %for.inc5 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi ptr [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, 11
  %sub = sub nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %sub
  store i32 %conv, ptr %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 45
  %0 = load i32, ptr %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.11, i64 1
  store i32 %0, ptr %B.addr.11, align 4
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body3, label %for.inc5

for.inc5:                                         ; preds = %for.body3
  %scevgep = getelementptr i32, ptr %B.addr.04, i64 10
  %inc6 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc6, 5
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end7

for.end7:                                         ; preds = %for.inc5
  ret void
}



;;  for (long int i = 0; i <= 5; i++)
;;    for (long int j = 0; j < 10; j++) {
;;      A[11*i - j] = i;
;;      *B++ = A[45];

define void @rdiv10(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - consistent input [S S]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc5
  %B.addr.04 = phi ptr [ %B, %entry ], [ %scevgep, %for.inc5 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc6, %for.inc5 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi ptr [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, 11
  %sub = sub nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %sub
  store i32 %conv, ptr %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 45
  %0 = load i32, ptr %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.11, i64 1
  store i32 %0, ptr %B.addr.11, align 4
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body3, label %for.inc5

for.inc5:                                         ; preds = %for.body3
  %scevgep = getelementptr i32, ptr %B.addr.04, i64 10
  %inc6 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc6, 6
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end7

for.end7:                                         ; preds = %for.inc5
  ret void
}


;;  for (long int i = 0; i < 5; i++)
;;    for (long int j = 0; j <= 10; j++) {
;;      A[11*i - j] = i;
;;      *B++ = A[45];

define void @rdiv11(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - consistent input [S S]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc5
  %B.addr.04 = phi ptr [ %B, %entry ], [ %scevgep, %for.inc5 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc6, %for.inc5 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi ptr [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, 11
  %sub = sub nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %sub
  store i32 %conv, ptr %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 45
  %0 = load i32, ptr %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.11, i64 1
  store i32 %0, ptr %B.addr.11, align 4
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 11
  br i1 %exitcond, label %for.body3, label %for.inc5

for.inc5:                                         ; preds = %for.body3
  %scevgep = getelementptr i32, ptr %B.addr.04, i64 11
  %inc6 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc6, 5
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end7

for.end7:                                         ; preds = %for.inc5
  ret void
}


;;  for (long int i = 0; i <= 5; i++)
;;    for (long int j = 0; j <= 10; j++) {
;;      A[11*i - j] = i;
;;      *B++ = A[45];

define void @rdiv12(ptr %A, ptr %B) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: da analyze - none!
; CHECK: da analyze - flow [* *|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - consistent input [S S]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc5
  %B.addr.04 = phi ptr [ %B, %entry ], [ %scevgep, %for.inc5 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc6, %for.inc5 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi ptr [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %conv = trunc i64 %i.03 to i32
  %mul = mul nsw i64 %i.03, 11
  %sub = sub nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %sub
  store i32 %conv, ptr %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 45
  %0 = load i32, ptr %arrayidx4, align 4
  %incdec.ptr = getelementptr inbounds i32, ptr %B.addr.11, i64 1
  store i32 %0, ptr %B.addr.11, align 4
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 11
  br i1 %exitcond, label %for.body3, label %for.inc5

for.inc5:                                         ; preds = %for.body3
  %scevgep = getelementptr i32, ptr %B.addr.04, i64 11
  %inc6 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc6, 6
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end7

for.end7:                                         ; preds = %for.inc5
  ret void
}
