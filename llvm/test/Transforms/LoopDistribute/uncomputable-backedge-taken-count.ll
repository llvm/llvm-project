; RUN: opt -passes=loop-distribute -enable-loop-distribute -verify-loop-info -verify-dom-info -S \
; RUN:   < %s | FileCheck %s

; NOTE: The tests below use infinite loops to force unknown backedge-taken counts.
; Making the exit condition depend on a load would break current loop-distribute,
; because it requires all accesses to end up in either of the loops, but not both.

; TODO
; Can distribute with unknown backedge-taken count, because no runtime checks are
; required.
define void @unknown_btc_distribute_no_checks_needed(ptr noalias %a, ptr noalias %c, ptr noalias %d) {
; CHECK-LABEL: @unknown_btc_distribute_no_checks_needed(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %for.body
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i32 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, ptr %a, i32 %ind
  %loadA = load i32, ptr %arrayidxA, align 4

  %mulA = mul i32 %loadA, 10

  %add = add nuw nsw i32 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, ptr %a, i32 %add
  store i32 %mulA, ptr %arrayidxA_plus_4, align 4

  %arrayidxD = getelementptr inbounds i32, ptr %d, i32 %ind
  %loadD = load i32, ptr %arrayidxD, align 4

  %mulC = mul i32 %loadD, 20

  %arrayidxC = getelementptr inbounds i32, ptr %c, i32 %ind
  store i32 %mulC, ptr %arrayidxC, align 4

  br i1 false, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Cannot distribute with unknown backedge-taken count, because runtime checks for
; induction wrapping are required.
define void @unknown_btc_do_not_distribute_wrapping_checks(ptr noalias %a, ptr noalias %c, ptr noalias %d) {
; CHECK-LABEL: @unknown_btc_do_not_distribute_wrapping_checks(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %for.body
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i32 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, ptr %a, i32 %ind
  %loadA = load i32, ptr %arrayidxA, align 4

  %mulA = mul i32 %loadA, 10

  %add = add i32 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, ptr %a, i32 %add
  store i32 %mulA, ptr %arrayidxA_plus_4, align 4

  %arrayidxD = getelementptr inbounds i32, ptr %d, i32 %ind
  %loadD = load i32, ptr %arrayidxD, align 4

  %mulC = mul i32 %loadD, 20

  %arrayidxC = getelementptr inbounds i32, ptr %c, i32 %ind
  store i32 %mulC, ptr %arrayidxC, align 4

  br i1 false, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
