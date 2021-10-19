; RUN: opt -S  -mtriple=s390x-unknown-linux -mcpu=z13 -loop-vectorize -dce \
; RUN:   -instcombine -force-vector-width=2  < %s | FileCheck %s
;
; Test that loop vectorizer does not generate vector addresses that must then
; always be extracted.

; Check that the addresses for a scalarized memory access is not extracted
; from a vector register.
define i32 @foo(i32* nocapture %A) {
;CHECK-LABEL: @foo(
;CHECK:  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
;CHECK:  %0 = shl nsw i64 %index, 2
;CHECK:  %1 = shl i64 %index, 2
;CHECK:  %2 = or i64 %1, 4
;CHECK:  %3 = getelementptr inbounds i32, i32* %A, i64 %0
;CHECK:  %4 = getelementptr inbounds i32, i32* %A, i64 %2
;CHECK:  store i32 4, i32* %3, align 4
;CHECK:  store i32 4, i32* %4, align 4

entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = shl nsw i64 %indvars.iv, 2
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %0
  store i32 4, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 undef
}


; Check that a load of address is scalarized.
define i32 @foo1(i32* nocapture noalias %A, i32** nocapture %PtrPtr) {
;CHECK-LABEL: @foo1(
;CHECK:  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
;CHECK:  %0 = or i64 %index, 1
;CHECK:  %1 = getelementptr inbounds i32*, i32** %PtrPtr, i64 %index
;CHECK:  %2 = getelementptr inbounds i32*, i32** %PtrPtr, i64 %0
;CHECK:  %3 = load i32*, i32** %1, align 8
;CHECK:  %4 = load i32*, i32** %2, align 8
;CHECK:  %5 = load i32, i32* %3, align 4
;CHECK:  %6 = load i32, i32* %4, align 4
;CHECK:  %7 = insertelement <2 x i32> poison, i32 %5, i32 0
;CHECK:  %8 = insertelement <2 x i32> %7, i32 %6, i32 1
;CHECK:  %9 = getelementptr inbounds i32, i32* %A, i64 %index
;CHECK:  %10 = bitcast i32* %9 to <2 x i32>*
;CHECK:  store <2 x i32> %8, <2 x i32>* %10, align 4

entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %ptr = getelementptr inbounds i32*, i32** %PtrPtr, i64 %indvars.iv
  %el = load i32*, i32** %ptr
  %v = load i32, i32* %el
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %v, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 undef
}
