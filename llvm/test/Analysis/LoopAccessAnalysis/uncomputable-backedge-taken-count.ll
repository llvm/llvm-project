; RUN: opt -passes='print-access-info' -aa-pipeline='basic-aa' -disable-output < %s  2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; TODO: Loop iteration counts are only required if we generate memory
;       runtime checks. Missing iteration counts should not prevent
;       analysis, if no runtime checks are required.

; No memory checks are required, because base pointers do not alias and we have
; a forward dependence for %a.
define void @safe_forward_dependence(i16* noalias %a,
                                     i16* noalias %b) {
; CHECK-LABEL: safe_forward_dependence
; CHECK:       for.body:
; CHECK-NEXT:     Report: could not determine number of loop iterations
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]

  %iv.next = add nuw nsw i64 %iv, 1

  %arrayidxA_plus_2 = getelementptr inbounds i16, i16* %a, i64 %iv.next
  %loadA_plus_2 = load i16, i16* %arrayidxA_plus_2, align 2

  %arrayidxB = getelementptr inbounds i16, i16* %b, i64 %iv
  %loadB = load i16, i16* %arrayidxB, align 2


  %mul = mul i16 %loadB, %loadA_plus_2

  %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %iv
  store i16 %mul, i16* %arrayidxA, align 2

  %exitcond = icmp eq i16 %loadB, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}




define void @unsafe_backwards_dependence(i16* noalias %a,
                                         i16* noalias %b) {
; CHECK-LABEL: unsafe_backwards_dependence
; CHECK:       for.body:
; CHECK-NEXT:     Report: could not determine number of loop iterations
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iv = phi i64 [ 1, %entry ], [ %iv.next, %for.body ]

  %idx = add nuw nsw i64 %iv, -1
  %iv.next = add nuw nsw i64 %iv, 1

  %arrayidxA_plus_2 = getelementptr inbounds i16, i16* %a, i64 %idx
  %loadA_plus_2 = load i16, i16* %arrayidxA_plus_2, align 2

  %arrayidxB = getelementptr inbounds i16, i16* %b, i64 %iv
  %loadB = load i16, i16* %arrayidxB, align 2


  %mul = mul i16 %loadB, %loadA_plus_2

  %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %iv
  store i16 %mul, i16* %arrayidxA, align 2

  %exitcond = icmp eq i16 %loadB, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}


define void @ptr_may_alias(i16* %a, i16* %b) {
; CHECK-LABEL: ptr_may_alias
; CHECK:       for.body:
; CHECK-NEXT:     Report: could not determine number of loop iterations
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iv = phi i64 [ 1, %entry ], [ %iv.next, %for.body ]

  %idx = add nuw nsw i64 %iv, -1
  %iv.next = add nuw nsw i64 %iv, 1

  %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %iv
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxB = getelementptr inbounds i16, i16* %b, i64 %iv
  %loadB = load i16, i16* %arrayidxB, align 2

  %mul = mul i16 %loadB, %loadA

  store i16 %mul, i16* %arrayidxA, align 2

  %exitcond = icmp eq i16 %loadB, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
