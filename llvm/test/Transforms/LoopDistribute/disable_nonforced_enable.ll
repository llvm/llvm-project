; RUN: opt -passes=loop-distribute -verify-loop-info -verify-dom-info -S < %s | FileCheck %s
;
; Check that llvm.loop.distribute.enable overrides
; llvm.loop.disable_nonforced.
;

define void @disable_nonforced(ptr noalias %a, ptr noalias %b, ptr noalias %c, ptr noalias %d,  ptr noalias %e) {
; CHECK-LABEL: @disable_nonforced(
; CHECK: for.body.ldist1:
entry:
  br label %for.body

for.body:
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, ptr %a, i64 %ind
  %loadA = load i32, ptr %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, ptr %b, i64 %ind
  %loadB = load i32, ptr %arrayidxB, align 4

  %mulA = mul i32 %loadB, %loadA

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, ptr %a, i64 %add
  store i32 %mulA, ptr %arrayidxA_plus_4, align 4

  %arrayidxD = getelementptr inbounds i32, ptr %d, i64 %ind
  %loadD = load i32, ptr %arrayidxD, align 4

  %arrayidxE = getelementptr inbounds i32, ptr %e, i64 %ind
  %loadE = load i32, ptr %arrayidxE, align 4

  %mulC = mul i32 %loadD, %loadE

  %arrayidxC = getelementptr inbounds i32, ptr %c, i64 %ind
  store i32 %mulC, ptr %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

!0 = distinct !{!0, !{!"llvm.loop.disable_nonforced"}, !{!"llvm.loop.distribute.enable", i32 1}}
