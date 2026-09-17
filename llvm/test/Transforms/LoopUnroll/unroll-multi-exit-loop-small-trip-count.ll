; RUN: opt < %s -passes=loop-unroll -unroll-runtime=true -verify-dom-info -verify-loop-info -unroll-runtime-other-exit-predictable=false -predictable-branch-threshold=99 -S | FileCheck %s
; RUN: opt < %s -passes=loop-unroll -unroll-runtime=true -verify-dom-info -verify-loop-info -unroll-runtime-other-exit-predictable=false -unroll-runtime-multi-exit-min-bodies=4 -predictable-branch-threshold=99 -S | FileCheck %s -check-prefix=MINBODIES4
; RUN: opt < %s -passes=loop-unroll -unroll-runtime=true -verify-dom-info -verify-loop-info -unroll-runtime-other-exit-predictable=false -unroll-runtime-multi-exit-min-bodies=0 -predictable-branch-threshold=99 -S | FileCheck %s -check-prefix=MINBODIES0

; Every function below is the same backward linear scan.

; size=200: expected scan of ~96 elements, P = 1/97 = 0.0103.
;
; This is the case the change was written for. It misses the fixed 1/100
; threshold by 3% and so was not unrolled, even though ~97 iterations leave
; room for an 8x body.
define i32 @scan_p97_size_200(ptr nocapture %a, i64 %n, i32 %key) {
; CHECK-LABEL: @scan_p97_size_200(
; CHECK:         header:
; CHECK-COUNT-8:   icmp eq i32 %{{.*}}, %key
; CHECK:         header.epil:
; CHECK:           icmp eq i32 %{{.*}}, %key
;
; MINBODIES4-LABEL: @scan_p97_size_200(
; MINBODIES4:         header:
; MINBODIES4-COUNT-8:   icmp eq i32 %{{.*}}, %key
;
; MINBODIES0-LABEL: @scan_p97_size_200(
; MINBODIES0-NOT: epil
;
entry:
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %arrayidx, align 4
  %match = icmp eq i32 %v, %key
  br i1 %match, label %found, label %latch, !prof !0

latch:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %notfound, label %header

notfound:
  ret i32 -1

found:
  %idx = trunc i64 %i to i32
  ret i32 %idx
}

!0 = !{!"branch_weights", i32 1, i32 96}

; size=20: expected scan of ~10 elements, P = 1/11 = 0.0909.
;
; The marginal case. 1/P = 11 clears the default requirement of 8 but only just,
; so the unrolled body runs about once before control falls into the epilogue.
; Requiring four bodies (1/32) rejects it.
define i32 @scan_p11_size_20(ptr nocapture %a, i64 %n, i32 %key) {
; CHECK-LABEL: @scan_p11_size_20(
; CHECK:         header:
; CHECK-COUNT-8:   icmp eq i32 %{{.*}}, %key
;
; MINBODIES4-LABEL: @scan_p11_size_20(
; MINBODIES4-NOT: epil
;
; MINBODIES0-LABEL: @scan_p11_size_20(
; MINBODIES0-NOT: epil
;
entry:
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %arrayidx, align 4
  %match = icmp eq i32 %v, %key
  br i1 %match, label %found, label %latch, !prof !1

latch:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %notfound, label %header

notfound:
  ret i32 -1

found:
  %idx = trunc i64 %i to i32
  ret i32 %idx
}

!1 = !{!"branch_weights", i32 1, i32 10}

; size=10: expected scan of ~5 elements, P = 1/6 = 0.167.
;
; Deliberately still rejected, at every setting.
define i32 @scan_p6_size_10(ptr nocapture %a, i64 %n, i32 %key) {
; CHECK-LABEL: @scan_p6_size_10(
; CHECK-NOT: epil
;
; MINBODIES4-LABEL: @scan_p6_size_10(
; MINBODIES4-NOT: epil
;
; MINBODIES0-LABEL: @scan_p6_size_10(
; MINBODIES0-NOT: epil
;
entry:
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %arrayidx, align 4
  %match = icmp eq i32 %v, %key
  br i1 %match, label %found, label %latch, !prof !2

latch:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %notfound, label %header

notfound:
  ret i32 -1

found:
  %idx = trunc i64 %i to i32
  ret i32 %idx
}

!2 = !{!"branch_weights", i32 1, i32 5}

; size=20 scan again, but forced to unroll by 16 instead of 8.
;
; Identical profile to @scan_p11_size_20, opposite decision. The requirement
; is now 1/P >= 16 and 1/P = 11. This is the property a fixed probability
; threshold cannot express at all.
define i32 @scan_p11_unroll_16(ptr nocapture %a, i64 %n, i32 %key) {
; CHECK-LABEL: @scan_p11_unroll_16(
; CHECK-NOT: epil
;
; MINBODIES4-LABEL: @scan_p11_unroll_16(
; MINBODIES4-NOT: epil
;
; MINBODIES0-LABEL: @scan_p11_unroll_16(
; MINBODIES0-NOT: epil
;
entry:
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %arrayidx, align 4
  %match = icmp eq i32 %v, %key
  br i1 %match, label %found, label %latch, !prof !1

latch:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %notfound, label %header, !llvm.loop !3

notfound:
  ret i32 -1

found:
  %idx = trunc i64 %i to i32
  ret i32 %idx
}

!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.unroll.count", i32 16}
