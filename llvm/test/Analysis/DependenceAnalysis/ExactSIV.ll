; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s
; RUN: opt < %s -analyze -enable-new-pm=0 -basic-aa -da | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"


;;  for (long unsigned i = 0; i < 10; i++) {
;;    A[i + 10] = i;
;;    *B++ = A[2*i + 1];

define void @exact0(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact0
; CHECK: da analyze - none!
; CHECK: da analyze - flow [<=|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %add = add i64 %i.02, 10
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %add
  store i32 %conv, i32* %arrayidx, align 4
  %mul = shl i64 %i.02, 1
  %add13 = or i64 %mul, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %add13
  %0 = load i32, i32* %arrayidx2, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i < 10; i++) {
;;    A[4*i + 10] = i;
;;    *B++ = A[2*i + 1];

define void @exact1(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact1
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
  %mul = shl i64 %i.02, 2
  %add = add i64 %mul, 10
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %add
  store i32 %conv, i32* %arrayidx, align 4
  %mul1 = shl i64 %i.02, 1
  %add23 = or i64 %mul1, 1
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i64 %add23
  %0 = load i32, i32* %arrayidx3, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i < 10; i++) {
;;    A[6*i] = i;
;;    *B++ = A[i + 60];

define void @exact2(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact2
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
  %mul = mul i64 %i.02, 6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %add = add i64 %i.02, 60
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %add
  %0 = load i32, i32* %arrayidx1, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i <= 10; i++) {
;;    A[6*i] = i;
;;    *B++ = A[i + 60];

define void @exact3(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact3
; CHECK: da analyze - none!
; CHECK: da analyze - flow [>]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul i64 %i.02, 6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %add = add i64 %i.02, 60
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %add
  %0 = load i32, i32* %arrayidx1, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 11
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i < 12; i++) {
;;    A[6*i] = i;
;;    *B++ = A[i + 60];

define void @exact4(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact4
; CHECK: da analyze - none!
; CHECK: da analyze - flow [>]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul i64 %i.02, 6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %add = add i64 %i.02, 60
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %add
  %0 = load i32, i32* %arrayidx1, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 12
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i <= 12; i++) {
;;    A[6*i] = i;
;;    *B++ = A[i + 60];

define void @exact5(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact5
; CHECK: da analyze - none!
; CHECK: da analyze - flow [=>|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul i64 %i.02, 6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %add = add i64 %i.02, 60
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %add
  %0 = load i32, i32* %arrayidx1, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 13
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i < 18; i++) {
;;    A[6*i] = i;
;;    *B++ = A[i + 60];

define void @exact6(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact6
; CHECK: da analyze - none!
; CHECK: da analyze - flow [=>|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul i64 %i.02, 6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %add = add i64 %i.02, 60
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %add
  %0 = load i32, i32* %arrayidx1, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 18
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i <= 18; i++) {
;;    A[6*i] = i;
;;    *B++ = A[i + 60];

define void @exact7(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact7
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
  %mul = mul i64 %i.02, 6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %add = add i64 %i.02, 60
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %add
  %0 = load i32, i32* %arrayidx1, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 19
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i < 10; i++) {
;;    A[-6*i] = i;
;;    *B++ = A[-i - 60];

define void @exact8(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact8
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
  %mul = mul i64 %i.02, -6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %sub1 = sub i64 -60, %i.02
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %sub1
  %0 = load i32, i32* %arrayidx2, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i <= 10; i++) {
;;    A[-6*i] = i;
;;    *B++ = A[-i - 60];

define void @exact9(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact9
; CHECK: da analyze - none!
; CHECK: da analyze - flow [>]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul i64 %i.02, -6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %sub1 = sub i64 -60, %i.02
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %sub1
  %0 = load i32, i32* %arrayidx2, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 11
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i < 12; i++) {
;;    A[-6*i] = i;
;;    *B++ = A[-i - 60];

define void @exact10(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact10
; CHECK: da analyze - none!
; CHECK: da analyze - flow [>]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul i64 %i.02, -6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %sub1 = sub i64 -60, %i.02
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %sub1
  %0 = load i32, i32* %arrayidx2, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 12
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i <= 12; i++) {
;;    A[-6*i] = i;
;;    *B++ = A[-i - 60];

define void @exact11(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact11
; CHECK: da analyze - none!
; CHECK: da analyze - flow [=>|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul i64 %i.02, -6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %sub1 = sub i64 -60, %i.02
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %sub1
  %0 = load i32, i32* %arrayidx2, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 13
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i < 18; i++) {
;;    A[-6*i] = i;
;;    *B++ = A[-i - 60];

define void @exact12(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact12
; CHECK: da analyze - none!
; CHECK: da analyze - flow [=>|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %B.addr.01 = phi i32* [ %B, %entry ], [ %incdec.ptr, %for.body ]
  %conv = trunc i64 %i.02 to i32
  %mul = mul i64 %i.02, -6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %sub1 = sub i64 -60, %i.02
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %sub1
  %0 = load i32, i32* %arrayidx2, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 18
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}


;;  for (long unsigned i = 0; i <= 18; i++) {
;;    A[-6*i] = i;
;;    *B++ = A[-i - 60];

define void @exact13(i32* %A, i32* %B) nounwind uwtable ssp {
entry:
  br label %for.body

; CHECK-LABEL: exact13
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
  %mul = mul i64 %i.02, -6
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  store i32 %conv, i32* %arrayidx, align 4
  %sub1 = sub i64 -60, %i.02
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %sub1
  %0 = load i32, i32* %arrayidx2, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %B.addr.01, i64 1
  store i32 %0, i32* %B.addr.01, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, 19
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}
