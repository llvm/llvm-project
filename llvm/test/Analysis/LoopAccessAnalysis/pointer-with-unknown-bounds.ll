; RUN: opt -loop-accesses -analyze < %s | FileCheck %s
; RUN: opt -passes='require<scalar-evolution>,require<aa>,loop(print-access-info)' -disable-output  < %s 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; We shouldn't quit the analysis if we encounter a pointer without known
; bounds *unless* we actually need to emit a memcheck for it.  (We only
; compute bounds for SCEVAddRecs so A[i*i] is deemed not having known bounds.)
;
; for (i = 0; i < 20; ++i)
;   A[i*i] *= 2;

; CHECK: for.body:
; CHECK:     Report: unsafe dependent memory operations in loop
; CHECK-NOT: Report: cannot identify array bounds
; CHECK:     Dependences:
; CHECK:       Unknown:
; CHECK:           %loadA = load i16, i16* %arrayidxA, align 2 ->
; CHECK:           store i16 %mul, i16* %arrayidxA, align 2

define void @f(i16* %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %access_ind = mul i64 %ind, %ind

  %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %access_ind
  %loadA = load i16, i16* %arrayidxA, align 2

  %mul = mul i16 %loadA, 2

  store i16 %mul, i16* %arrayidxA, align 2

  %add = add nuw nsw i64 %ind, 1
  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
