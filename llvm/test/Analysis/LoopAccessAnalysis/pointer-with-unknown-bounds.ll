; RUN: opt -aa-pipeline=basic-aa -passes='print-access-info' -disable-output  < %s 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; We shouldn't quit the analysis if we encounter a pointer without known
; bounds *unless* we actually need to emit a memcheck for it.  (We only
; compute bounds for SCEVAddRecs so A[i*i] is deemed not having known bounds.)
;
; for (i = 0; i < 20; ++i)
;   A[i*i] *= 2;

; CHECK-LABEL: addrec_squared
; CHECK-NEXT: for.body:
; CHECK-NEXT:   Report: unsafe dependent memory operations in loop
; CHECK-NOT:    Report: cannot identify array bounds
; CHECK-NEXT:   Unknown data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       Unknown:
; CHECK-NEXT:         %loadA = load i16, i16* %arrayidxA, align 2 ->
; CHECK-NEXT:         store i16 %mul, i16* %arrayidxA, align 2

define void @addrec_squared(i16* %a) {
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

; TODO: We cannot compute the bound for %arrayidxA_ub, because the index is
; loaded on each iteration. As %a and %b are no-alias, no memchecks are required
; and unknown bounds should not prevent further analysis.
define void @loaded_bound(i16* noalias %a, i16* noalias %b) {
; CHECK-LABEL: loaded_bound
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Report: cannot identify array bounds
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]

  %iv.next = add nuw nsw i64 %iv, 1

  %arrayidxB = getelementptr inbounds i16, i16* %b, i64 %iv
  %loadB = load i16, i16* %arrayidxB, align 2

  %arrayidxA_ub = getelementptr inbounds i16, i16* %a, i16 %loadB
  %loadA_ub = load i16, i16* %arrayidxA_ub, align 2

  %mul = mul i16 %loadB, %loadA_ub

  %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %iv
  store i16 %mul, i16* %arrayidxA, align 2

  %exitcond = icmp eq i64 %iv, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
