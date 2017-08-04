; RUN: opt -basicaa -aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

; Check that BasicAA falls back to MayAlias (instead of PartialAlias) when none
; of its little tricks are applicable.

; CHECK: MayAlias: float* %arrayidxA, float* %arrayidxB

define void @fallback_mayalias(float* noalias nocapture %C, i64 %i, i64 %j) local_unnamed_addr {
entry:
  %shl = shl i64 %i, 3
  %mul = shl nsw i64 %j, 4
  %addA = add nsw i64 %mul, %shl
  %orB = or i64 %shl, 1
  %addB = add nsw i64 %mul, %orB

  %arrayidxA = getelementptr inbounds float, float* %C, i64 %addA
  store float undef, float* %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds float, float* %C, i64 %addB
  store float undef, float* %arrayidxB, align 4

  ret void
}
