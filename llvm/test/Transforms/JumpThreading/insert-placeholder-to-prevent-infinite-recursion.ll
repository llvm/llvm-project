; RUN: opt -jump-threading -S < %s | FileCheck %s
@a = common dso_local local_unnamed_addr global i16 0, align 2

; Function Attrs: nofree norecurse nounwind
define internal fastcc void @s() unnamed_addr {
; CHECK-LABEL: @s(
; CHECK-NEXT:  for.cond1.preheader.lr.ph:
; CHECK-NEXT:    br label [[FOR_COND1_PREHEADER:%.*]]
; CHECK:       for.cond1.preheader:
; CHECK-NEXT:    [[DOTPR_I_19:%.*]] = phi i32 [ undef, [[FOR_COND1_PREHEADER_LR_PH:%.*]] ], [ 0, [[FOR_INC_SPLIT_1:%.*]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = phi i16 [ undef, [[FOR_COND1_PREHEADER_LR_PH]] ], [ [[INC9:%.*]], [[FOR_INC_SPLIT_1]] ]
; CHECK-NEXT:    [[TOBOOL4_I:%.*]] = icmp eq i32 [[DOTPR_I_19]], 0
; CHECK-NEXT:    br i1 [[TOBOOL4_I]], label [[FOR_INC_SPLIT_1]], label [[FOR_BODY_LR_PH_I:%.*]]
; CHECK:       for.body.lr.ph.i:
; CHECK-NEXT:    ret void
; CHECK:       for.cond.for.end10_crit_edge:
; CHECK-NEXT:    ret void
; CHECK:       for.inc.split.1:
; CHECK-NEXT:    [[INC9]] = add i16 [[TMP0]], 1
; CHECK-NEXT:    [[TOBOOL:%.*]] = icmp eq i16 [[INC9]], 0
; CHECK-NEXT:    br i1 [[TOBOOL]], label [[FOR_COND_FOR_END10_CRIT_EDGE:%.*]], label [[FOR_COND1_PREHEADER]]
;
for.cond1.preheader.lr.ph:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc.split.1, %for.cond1.preheader.lr.ph
  %.pr.i.19 = phi i32 [ undef, %for.cond1.preheader.lr.ph ], [ 0, %for.inc.split.1 ]
  %0 = phi i16 [ undef, %for.cond1.preheader.lr.ph ], [ %inc9, %for.inc.split.1 ]
  %tobool4.i = icmp eq i32 %.pr.i.19, 0
  br i1 %tobool4.i, label %t.exit, label %for.body.lr.ph.i

for.body.lr.ph.i:                                 ; preds = %for.cond1.preheader
  ret void

t.exit:                                           ; preds = %for.cond1.preheader
  br label %for.inc.split

for.inc.split:                                    ; preds = %t.exit
  %tobool4.i.1 = icmp eq i32 %.pr.i.19, 0
  br i1 %tobool4.i.1, label %for.inc.split.1, label %for.body.lr.ph.i.1

for.cond.for.end10_crit_edge:                     ; preds = %for.inc.split.1
  ret void

for.body.lr.ph.i.1:                               ; preds = %for.inc.split
  br label %for.body.lr.ph.split.i.1

for.body.lr.ph.split.i.1:                         ; preds = %for.body.lr.ph.i.1
  br label %for.body.preheader.i.1

for.body.preheader.i.1:                           ; preds = %for.body.lr.ph.split.i.1
  br label %for.body.i.us.1.preheader

for.body.i.us.1.preheader:                        ; preds = %for.body.preheader.i.1
  br label %for.body.i.us.1

for.body.i.us.1:                                  ; preds = %lor.end.i.us.1, %for.body.i.us.1.preheader
  %.b.i.us.1 = phi i1 [ %spec.select44.i.us.1, %lor.end.i.us.1 ], [ undef, %for.body.i.us.1.preheader ]
  br i1 %.b.i.us.1, label %lor.end.i.us.1, label %lor.rhs.i.us.1

lor.rhs.i.us.1:                                   ; preds = %for.body.i.us.1
  %conv5.i.us.1 = trunc i32 undef to i8
  br label %lor.end.i.us.1

lor.end.i.us.1:                                   ; preds = %lor.rhs.i.us.1, %for.body.i.us.1
  %.b31.i.us.1 = or i1 undef, %.b.i.us.1
  %spec.select44.i.us.1 = or i1 undef, %.b31.i.us.1
  br i1 undef, label %for.end18.loopexit42.i.1.loopexit, label %for.body.i.us.1

for.end18.loopexit42.i.1.loopexit:                ; preds = %lor.end.i.us.1
  br label %for.end18.loopexit42.i.1

for.end18.loopexit42.i.1:                         ; preds = %for.end18.loopexit42.i.1.loopexit
  br label %for.inc.split.1

for.inc.split.1:                                  ; preds = %for.end18.loopexit42.i.1, %for.inc.split
  %inc9 = add i16 %0, 1
  %tobool = icmp eq i16 %inc9, 0
  br i1 %tobool, label %for.cond.for.end10_crit_edge, label %for.cond1.preheader
}
