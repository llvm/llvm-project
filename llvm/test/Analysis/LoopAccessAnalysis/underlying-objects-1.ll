; RUN: opt -passes='print-access-info' -disable-output  < %s 2>&1 | FileCheck %s

; In:
;
;   store_ptr = A;
;   load_ptr = &A[2];
;   for (i = 0; i < n; i++)
;    *store_ptr++ = *load_ptr++ *10;  // A[i] = A[i+2] * 10
;
; make sure, we look through the PHI to conclude that store_ptr and load_ptr
; both have A as their underlying object.  The dependence is safe for
; vectorization requiring no memchecks.
;
; Otherwise we would try to prove independence with a memcheck that is going
; to always fail.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; CHECK: Memory dependences are safe{{$}}

define void @f(i8* noalias %A, i64 %width) {
for.body.preheader:
  %A_ahead = getelementptr inbounds i8, i8* %A, i64 2
  br label %for.body

for.body:
  %i = phi i64 [ %i.1, %for.body ], [ 0, %for.body.preheader ]
  %load_ptr = phi i8* [ %load_ptr.1, %for.body ], [ %A_ahead, %for.body.preheader ]
  %store_ptr = phi i8* [ %store_ptr.1, %for.body ], [ %A, %for.body.preheader ]

  %loadA = load i8, i8* %load_ptr, align 1

  %mul = mul i8 %loadA, 10

  store i8 %mul, i8* %store_ptr, align 1

  %load_ptr.1 = getelementptr inbounds i8, i8* %load_ptr, i64 1
  %store_ptr.1 = getelementptr inbounds i8, i8* %store_ptr, i64 1
  %i.1 = add nuw i64 %i, 1

  %exitcond = icmp eq i64 %i.1, %width
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
