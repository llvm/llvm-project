; RUN: opt -passes='print<access-info>' -disable-output  < %s 2>&1 | FileCheck %s

; Analyze this loop:
;   for (i = 0; i < n; i++)
;    A[i + 4] = A[i] * 2;

; CHECK: Memory dependences are safe with a maximum safe vector width of 64 bits

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

@A = common global ptr null, align 8

define void @f() {
entry:
  %a = load ptr, ptr @A, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i16, ptr %a, i64 %ind
  %loadA = load i16, ptr %arrayidxA, align 2

  %mul = mul i16 %loadA, 2

  %next = add nuw nsw i64 %ind, 4
  %arrayidxA_next = getelementptr inbounds i16, ptr %a, i64 %next
  store i16 %mul, ptr %arrayidxA_next, align 2

  %add = add nuw nsw i64 %ind, 1
  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
