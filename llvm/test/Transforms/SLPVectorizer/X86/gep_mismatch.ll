; RUN: opt < %s -S -slp-vectorizer

; This code has GEPs with different index types, which should not
; matter for the SLPVectorizer.

target triple = "x86_64--linux"

define void @foo() {
entry:
  br label %bb1

bb1:
  %ls1.ph = phi float* [ %_tmp1, %bb1 ], [ undef, %entry ]
  %ls2.ph = phi float* [ %_tmp2, %bb1 ], [ undef, %entry ]
  store float undef, float* %ls1.ph
  %_tmp1 = getelementptr float, float* %ls1.ph, i32 1
  %_tmp2 = getelementptr float, float* %ls2.ph, i64 4
  br i1 false, label %bb1, label %bb2

bb2:
  ret void
}
