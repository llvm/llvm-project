; RUN: opt < %s -passes='print<scalar-evolution>,loop(loop-rotate),invalidate<scalar-evolution>,print<scalar-evolution>' -disable-output 2>&1 | FileCheck -check-prefixes CHECK-SCEV,CHECK-SCEV-OK %s
; RUN: opt < %s -passes='print<scalar-evolution>,loop(loop-rotate),print<scalar-evolution>' -disable-output 2>&1 | FileCheck -check-prefixes CHECK-SCEV,CHECK-SCEV-NOK %s
; FIXME (crashes): opt < %s -passes='loop(canon-freeze),loop(loop-rotate),print<scalar-evolution>' -disable-output

; In the first two RUN lines print<scalar-evolution> is used to populate the
; analysis cache before loop-rotate. That seem to be enough to see the problem
; by examining print<scalar-evolution> printouts after loop-rotate. However,
; I've only seen the crashes when using canon-freeze as a trigger to populate
; the analysis cache.

; Verify that we get the same SCEV expressions after loop-rotate, regardless if we invalidate scalar-evolution before the final printing or not.
; FIXME: As indicated by CHECK-SCEV-OK vs CHECK-SCEV-NOK this isn't currently true (PR51981).
;
; CHECK-SCEV: Classifying expressions for: @test_function
; CHECK-SCEV:   %wide = load i32, i32* @offset, align 1
; CHECK-SCEV:   -->  %wide U: full-set S: full-set          Exits: <<Unknown>>              LoopDispositions: { %loop.outer.header: Variant, %loop.inner: Invariant }
; CHECK-SCEV:   %narrow = trunc i32 %wide to i16
; CHECK-SCEV:   -->  (trunc i32 %wide to i16) U: full-set S: full-set               Exits: <<Unknown>>              LoopDispositions: { %loop.outer.header: Variant, %loop.inner: Invariant }
; CHECK-SCEV:   %iv = phi i16 [ %narrow, %loop.inner.ph ], [ %iv.plus, %loop.inner ]
; CHECK-SCEV:   -->  {(trunc i32 %wide to i16),+,1}<%loop.inner> U: full-set S: full-set           Exits: (-1 + (700 umax (1 + (trunc i32 %wide to i16))))               LoopDispositions: { %loop.inner: Computable, %loop.outer.header: Variant }
;
; CHECK-SCEV: Classifying expressions for: @test_function
; CHECK-SCEV:   %wide1 = load i32, i32* @offset, align 1
; CHECK-SCEV:   -->  %wide1 U: full-set S: full-set
; CHECK-SCEV:   %wide2 = phi i32 [ %wide1, %loop.inner.ph.lr.ph ], [ %wide, %loop.outer.latch ]
; CHECK-SCEV:   -->  %wide2 U: full-set S: full-set         Exits: <<Unknown>>              LoopDispositions: { %loop.inner.ph: Variant, %loop.inner: Invariant }
; CHECK-SCEV:   %narrow = trunc i32 %wide2 to i16
; CHECK-SCEV-OK:   -->  (trunc i32 %wide2 to i16) U: full-set S: full-set               Exits: <<Unknown>>              LoopDispositions: { %loop.inner.ph: Variant, %loop.inner: Invariant }
; CHECK-SCEV-NOK:   -->  (trunc i32 %wide to i16) U: full-set S: full-set               Exits: <<Unknown>>              LoopDispositions: { %loop.inner.ph: Variant, %loop.inner: Invariant }
; CHECK-SCEV:   %iv = phi i16 [ %narrow, %loop.inner.ph ], [ %iv.plus, %loop.inner ]
; CHECK-SCEV-OK:   -->  {(trunc i32 %wide2 to i16),+,1}<%loop.inner> U: full-set S: full-set           Exits: (-1 + (700 umax (1 + (trunc i32 %wide2 to i16))))               LoopDispositions: { %loop.inner: Computable, %loop.inner.ph: Variant }
; CHECK-SCEV-NOK:   -->  {(trunc i32 %wide to i16),+,1}<%loop.inner> U: full-set S: full-set           Exits: (-1 + (700 umax (1 + (trunc i32 %wide to i16))))               LoopDispositions: { %loop.inner: Computable, %loop.inner.ph: Variant }


@offset = external dso_local global i32, align 1
@array = internal global [11263 x i32] zeroinitializer, align 1

define void @test_function(i1 %cond) {
entry:
  br label %loop.outer.header

loop.outer.header:                                ; preds = %loop.outer.latch, %entry
  %wide = load i32, i32* @offset, align 1
  br i1 %cond, label %exit, label %loop.inner.ph

loop.inner.ph:                                    ; preds = %loop.outer.header
  %narrow = trunc i32 %wide to i16
  br label %loop.inner

loop.inner:                                       ; preds = %loop.inner, %loop.inner.ph
  %iv = phi i16 [ %narrow, %loop.inner.ph ], [ %iv.plus, %loop.inner ]
  %iv.promoted = zext i16 %iv to i32
  %gep = getelementptr inbounds [11263 x i32], [11263 x i32]* @array, i32 0, i32 %iv.promoted
  store i32 7, i32* %gep, align 1
  %iv.plus = add i16 %iv, 1
  %cmp = icmp ult i16 %iv.plus, 700
  br i1 %cmp, label %loop.inner, label %loop.outer.latch

loop.outer.latch:                                 ; preds = %loop.inner
  br label %loop.outer.header

exit:                                             ; preds = %loop.outer.header
  ret void
}

