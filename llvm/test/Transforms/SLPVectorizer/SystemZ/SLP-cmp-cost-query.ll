; REQUIRES: asserts
; RUN: opt -mtriple=systemz-unknown -mcpu=z13 -slp-vectorizer -debug-only=SLP \
; RUN:   -S -disable-output < %s 2>&1 | FileCheck %s
;
; Check that SLP vectorizer gets the right cost difference for a compare
; node.

; Function Attrs: norecurse nounwind readonly
define void @fun(i8* nocapture, i32 zeroext) local_unnamed_addr #0 {
.lr.ph.preheader:
  br label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.lr.ph
  %2 = phi i32 [ %., %.lr.ph ], [ undef, %.lr.ph.preheader ]
  %3 = phi i32 [ %.9, %.lr.ph ], [ undef, %.lr.ph.preheader ]
  %4 = icmp ult i32 %2, %1
  %5 = select i1 %4, i32 0, i32 %1
  %. = sub i32 %2, %5
  %6 = icmp ult i32 %3, %1
  %7 = select i1 %6, i32 0, i32 %1
  %.9 = sub i32 %3, %7
  %8 = zext i32 %. to i64
  %9 = getelementptr inbounds i8, i8* %0, i64 %8
  %10 = load i8, i8* %9, align 1
  %11 = zext i32 %.9 to i64
  %12 = getelementptr inbounds i8, i8* %0, i64 %11
  %13 = load i8, i8* %12, align 1
  %14 = icmp eq i8 %10, %13
  br i1 %14, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph
  ret void

; CHECK: SLP: Adding cost -1 for bundle that starts with   %4 = icmp ult i32 %2, %1.
}

