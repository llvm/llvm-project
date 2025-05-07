; RUN: opt -passes=simplifycfg -S --preserve-ll-uselistorder %s | FileCheck %s
; REQUIRES: x86-registered-target
; CHECK-LABEL: @n
; CHECK: uselistorder i16 0, { 3, 2, 4, 1, 5, 0, 6 }

; Note: test was added in an effort to ensure determinism when updating memoryssa. See PR42574.
; If the uselistorder check becomes no longer relevant, the test can be disabled or removed.

%rec9 = type { i16, i32, i32 }

@a = global [1 x [1 x %rec9]] zeroinitializer

define i16 @n() {
  br label %..split_crit_edge

..split_crit_edge:                                ; preds = %0
  br label %.split

bb4.us4:                                          ; preds = %bb2.split.us32, %bb6.us28
  %i.4.01.us5 = phi i16 [ %_tmp49.us30, %bb6.us28 ]
  br label %g.exit4.us21

bb1.i.us14:                                       ; preds = %bb4.us4
  br label %g.exit4.us21

g.exit4.us21:                                     ; preds = %bb1.i.us14, %g.exit4.critedge.us9
  %i.4.02.us22 = phi i16 [ %i.4.01.us5, %bb4.us4 ], [ %i.4.01.us5, %bb1.i.us14 ]
  br label %bb6.us28

bb5.us26:                                         ; preds = %g.exit4.us21
  br label %bb6.us28

bb6.us28:                                         ; preds = %bb5.us26, %g.exit4.us21
  %i.4.03.us29 = phi i16 [ %i.4.02.us22, %bb5.us26 ], [ %i.4.02.us22, %g.exit4.us21 ]
  %_tmp49.us30 = add nuw nsw i16 %i.4.03.us29, 1
  br label %bb4.us4

bb4.us.us:                                        ; preds = %bb2.split.us.us, %bb6.us.us
  %i.4.01.us.us = phi i16  [ %_tmp49.us.us, %bb6.us.us ]
  br label %bb1.i.us.us

bb1.i.us.us:                                      ; preds = %bb4.us.us
  br label %g.exit4.us.us

g.exit4.us.us:                                    ; preds = %bb1.i.us.us, %g.exit4.critedge.us.us
  %i.4.02.us.us = phi i16 [ %i.4.01.us.us, %bb1.i.us.us ]
  br label %bb5.us.us

bb5.us.us:                                        ; preds = %g.exit4.us.us
  br label %bb6.us.us

bb6.us.us:                                        ; preds = %bb5.us.us, %g.exit4.us.us
  %i.4.03.us.us = phi i16 [ %i.4.02.us.us, %bb5.us.us ]
  %_tmp49.us.us = add nuw nsw i16 %i.4.03.us.us, 1
  br label %bb4.us.us


.split:                                           ; preds = %..split_crit_edge
  br label %bb2

bb2:                                              ; preds = %.split, %bb7
  %h.3.0 = phi i16 [ undef, %.split ], [ %_tmp53, %bb7 ]
  br label %bb2.bb2.split_crit_edge

bb2.bb2.split_crit_edge:                          ; preds = %bb2
  br label %bb2.split

bb2.split.us:                                     ; preds = %bb2
  br label %bb4.us

bb4.us:                                           ; preds = %bb6.us, %bb2.split.us
  %i.4.01.us = phi i16 [ 0, %bb2.split.us ]
  br label %bb1.i.us

g.exit4.critedge.us:                              ; preds = %bb4.us
  br label %g.exit4.us

bb1.i.us:                                         ; preds = %bb4.us
  br label %g.exit4.us

g.exit4.us:                                       ; preds = %bb1.i.us, %g.exit4.critedge.us
  %i.4.02.us = phi i16 [ %i.4.01.us, %g.exit4.critedge.us ], [ %i.4.01.us, %bb1.i.us ]
  br label %bb5.us

bb5.us:                                           ; preds = %g.exit4.us
  br label %bb7

bb2.split:                                        ; preds = %bb2.bb2.split_crit_edge
  br label %bb4

bb4:                                              ; preds = %bb2.split, %bb6
  %i.4.01 = phi i16 [ 0, %bb2.split ]
  %_tmp16 = getelementptr [1 x [1 x %rec9]], ptr @a, i16 0, i16 %h.3.0, i16 %i.4.01, i32 0
  %_tmp17 = load i16, ptr %_tmp16, align 1
  br label %g.exit4.critedge

bb1.i:                                            ; preds = %bb4
  br label %g.exit4

g.exit4.critedge:                                 ; preds = %bb4
  %_tmp28.c = getelementptr [1 x [1 x %rec9]], ptr @a, i16 0, i16 %h.3.0, i16 %i.4.01, i32 1
  %_tmp29.c = load i32, ptr %_tmp28.c, align 1
  %_tmp30.c = trunc i32 %_tmp29.c to i16
  br label %g.exit4

g.exit4:                                          ; preds = %g.exit4.critedge, %bb1.i
  %i.4.02 = phi i16 [ %i.4.01, %g.exit4.critedge ], [ %i.4.01, %bb1.i ]
  %_tmp41 = getelementptr [1 x [1 x %rec9]], ptr @a, i16 0, i16 %h.3.0, i16 %i.4.02, i32 2
  br label %bb6

bb5:                                              ; preds = %g.exit4
  br label %bb6

bb6:                                              ; preds = %bb5, %g.exit4
  %i.4.03 = phi i16 [ %i.4.02, %bb5 ], [ %i.4.02, %g.exit4 ]
  %_tmp49 = add nuw nsw i16 %i.4.03, 1
  br label %bb7

bb7:                                              ; preds = %bb7.us-lcssa.us, %bb7.us-lcssa
  %_tmp53 = add nsw i16 %h.3.0, 1
  br label %bb2
}
