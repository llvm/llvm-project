; RUN: opt -basic-aa -loop-accesses -analyze < %s | FileCheck %s
; RUN: opt -passes='require<aa>,require<scalar-evolution>,require<aa>,loop(print-access-info)' -aa-pipeline='basic-aa' -disable-output < %s  2>&1 | FileCheck %s

; For this loop:
;   for (int i = 0; i < n; i++)
;    A[2 * i] = A[2 * i] + B[i];
;
; , SCEV is unable to prove that A[2 * i] does not overflow.  However,
; analyzing the IR helps us to conclude it and in turn allow dependence
; analysis.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: Memory dependences are safe{{$}}

define void @f(i16* noalias %a,
               i16* noalias %b, i64 %N) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %inc, %for.body ]

  %mul = mul nuw nsw i64 %ind, 2

  %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %mul
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxB = getelementptr inbounds i16, i16* %b, i64 %ind
  %loadB = load i16, i16* %arrayidxB, align 2

  %add = mul i16 %loadA, %loadB

  store i16 %add, i16* %arrayidxA, align 2

  %inc = add nuw nsw i64 %ind, 1
  %exitcond = icmp eq i64 %inc, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
