; RUN: opt -passes='loop-mssa(simple-loop-unswitch<nontrivial>)' -S < %s | FileCheck %s

; Reduced from a csmith program compiled with instrumentation-based PGO.
; The inner loop (%bb8/%bb11) is entered 3900 times in the profile: on 300
; entries %i7 is true and the loop runs 5 iterations (1 + 1200/300), on 3600
; entries %i7 is false and the loop exits through %.loopexit on its first
; iteration.
;
; The invariant branch on %i7 exits the loop, so its in-loop weight (1500) was
; collected per iteration. Non-trivial unswitching hoists it into the
; preheader, where it runs once per loop entry, so the in-loop weight must be
; divided by the estimated trip count (5), giving 300. The exit weight (3600)
; is already a per-entry count and is kept.

@d = global i8 0

define i32 @main() !prof !0 {
; CHECK-LABEL: define i32 @main(
; CHECK:       .critedge:
; CHECK:         br i1 %{{.*}}, label %.critedge.split.us, label %.critedge.split, !prof [[PROF_HOISTED:![0-9]+]]
; CHECK:       bb8.us:
; CHECK:         br label %bb11.us
; CHECK:       bb11.us:
; CHECK:         br i1 %i13.us, label %._crit_edge.split.us, label %bb8.us, !prof [[PROF_LATCH:![0-9]+]]
bb:
  br label %bb1

bb1:                                              ; preds = %._crit_edge1, %bb
  %i = phi i32 [ 0, %bb ], [ %i17, %._crit_edge1 ]
  %.mask = and i32 %i, 1
  %i2 = icmp eq i32 %.mask, 0
  br i1 %i2, label %.critedge.preheader, label %bb3, !prof !1

bb3:                                              ; preds = %bb1
  %i4 = srem i32 1, %i
  %i5 = icmp eq i32 %i4, 0
  br i1 %i5, label %._crit_edge1, label %.critedge.preheader, !prof !2

.critedge.preheader:                              ; preds = %bb3, %bb1
  br label %.critedge

.critedge:                                        ; preds = %bb14, %.critedge.preheader
  %i6 = phi i16 [ %i15, %bb14 ], [ 0, %.critedge.preheader ]
  %i7 = icmp eq i32 %i, 0
  br label %bb8

bb8:                                              ; preds = %bb11, %.critedge
  %i9 = phi i32 [ 1, %.critedge ], [ %i12, %bb11 ]
  %i10 = load volatile i8, ptr @d, align 1
  br i1 %i7, label %bb11, label %.loopexit, !prof !3

bb11:                                             ; preds = %bb8
  %i12 = add i32 %i9, 1
  %i13 = icmp eq i32 %i12, 6
  br i1 %i13, label %._crit_edge, label %bb8, !prof !4, !llvm.loop !5

._crit_edge:                                      ; preds = %bb11
  br label %bb14, !llvm.loop !5

.loopexit:                                        ; preds = %bb8
  br label %bb14

bb14:                                             ; preds = %.loopexit, %._crit_edge
  %i15 = add i16 %i6, 1
  %i16 = icmp eq i16 %i15, 6
  br i1 %i16, label %._crit_edge1.loopexit, label %.critedge, !prof !8, !llvm.loop !9

._crit_edge1.loopexit:                            ; preds = %bb14
  br label %._crit_edge1

._crit_edge1:                                     ; preds = %._crit_edge1.loopexit, %bb3
  %i17 = add i32 %i, -1
  %i18 = icmp eq i32 %i17, -14
  br i1 %i18, label %bb19, label %bb1, !prof !10

bb19:                                             ; preds = %._crit_edge1
  ret i32 0
}

!0 = !{!"function_entry_count", i64 50}
!1 = !{!"branch_weights", i32 350, i32 350}
!2 = !{!"branch_weights", i32 50, i32 300}
!3 = !{!"branch_weights", i32 1500, i32 3600}
!4 = !{!"branch_weights", i32 300, i32 1200}
!5 = distinct !{!5, !6, !7}
!6 = !{!"llvm.loop.mustprogress"}
!7 = !{!"llvm.loop.unroll.disable"}
!8 = !{!"branch_weights", i32 650, i32 3250}
!9 = distinct !{!9, !6, !7}
!10 = !{!"branch_weights", i32 50, i32 650}

; CHECK: [[PROF_HOISTED]] = !{!"branch_weights", i32 300, i32 3600}
; CHECK: [[PROF_LATCH]] = !{!"branch_weights", i32 300, i32 1200}
