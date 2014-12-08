; RUN: opt < %s -indvars -S | FileCheck %s

@a = common global i32 0, align 4
@c = common global i32 0, align 4
@b = common global i32 0, align 4

define void @f() {
; CHECK-LABEL: @f(
; CHECK-LABEL: entry:
; CHECK: br label %[[for_cond2_preheader:.*]]

; CHECK: [[for_cond2_preheader]]:
; CHECK-NEXT: %[[indvars_iv:.*]] = phi i32 [ %[[indvars_iv_next:.*]], %[[for_inc13:.*]] ], [ -14, %entry ]
; br i1 {{.*}}, label %[[for_inc13]], label %
entry:
  %0 = load i32* @a, align 4
  %tobool2 = icmp eq i32 %0, 0
  %1 = load i32* @a, align 4
  %tobool = icmp eq i32 %1, 0
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.inc13, %entry
  %storemerge15 = phi i8 [ -14, %entry ], [ %inc14, %for.inc13 ]
  br i1 %tobool2, label %for.inc13, label %for.body3.lr.ph

for.body3.lr.ph:                                  ; preds = %for.cond2.preheader
  %tobool5 = icmp eq i8 %storemerge15, 0
  %conv7 = sext i8 %storemerge15 to i32
  %2 = add nsw i32 %conv7, 1
  %3 = icmp ult i32 %2, 3
  %div = select i1 %3, i32 %conv7, i32 0
  br i1 %tobool5, label %for.body3.lr.ph.split.us, label %for.body3.lr.ph.for.body3.lr.ph.split_crit_edge

for.body3.lr.ph.for.body3.lr.ph.split_crit_edge:  ; preds = %for.body3.lr.ph
  br label %for.body3.lr.ph.split

for.body3.lr.ph.split.us:                         ; preds = %for.body3.lr.ph
  br i1 %tobool, label %for.body3.lr.ph.split.us.split.us, label %for.body3.lr.ph.split.us.for.body3.lr.ph.split.us.split_crit_edge

for.body3.lr.ph.split.us.for.body3.lr.ph.split.us.split_crit_edge: ; preds = %for.body3.lr.ph.split.us
  br label %for.body3.lr.ph.split.us.split

for.body3.lr.ph.split.us.split.us:                ; preds = %for.body3.lr.ph.split.us
  br label %for.body3.us.us

for.body3.us.us:                                  ; preds = %for.cond2.loopexit.us.us, %for.body3.lr.ph.split.us.split.us
  br i1 true, label %cond.false.us.us, label %cond.end.us.us

cond.false.us.us:                                 ; preds = %for.body3.us.us
  br label %cond.end.us.us

cond.end.us.us:                                   ; preds = %cond.false.us.us, %for.body3.us.us
  %cond.us.us = phi i32 [ %div, %cond.false.us.us ], [ %conv7, %for.body3.us.us ]
  %4 = load i32* @b, align 4
  %cmp91.us.us = icmp slt i32 %4, 1
  br i1 %cmp91.us.us, label %for.inc.lr.ph.us.us, label %for.cond2.loopexit.us.us

for.cond2.loopexit.us.us:                         ; preds = %for.cond8.for.cond2.loopexit_crit_edge.us.us, %cond.end.us.us
  br i1 true, label %for.cond2.for.inc13_crit_edge.us-lcssa.us.us-lcssa.us, label %for.body3.us.us

for.inc.lr.ph.us.us:                              ; preds = %cond.end.us.us
  br label %for.inc.us.us

for.cond8.for.cond2.loopexit_crit_edge.us.us:     ; preds = %for.inc.us.us
  %inc.lcssa.us.us = phi i32 [ %inc.us.us, %for.inc.us.us ]
  store i32 %inc.lcssa.us.us, i32* @b, align 4
  br label %for.cond2.loopexit.us.us

for.inc.us.us:                                    ; preds = %for.inc.us.us, %for.inc.lr.ph.us.us
  %5 = phi i32 [ %4, %for.inc.lr.ph.us.us ], [ %inc.us.us, %for.inc.us.us ]
  %inc.us.us = add nsw i32 %5, 1
  %cmp9.us.us = icmp slt i32 %inc.us.us, 1
  br i1 %cmp9.us.us, label %for.inc.us.us, label %for.cond8.for.cond2.loopexit_crit_edge.us.us

for.cond2.for.inc13_crit_edge.us-lcssa.us.us-lcssa.us: ; preds = %for.cond2.loopexit.us.us
  %cond.lcssa.ph.us.ph.us = phi i32 [ %cond.us.us, %for.cond2.loopexit.us.us ]
  br label %for.cond2.for.inc13_crit_edge.us-lcssa.us

for.body3.lr.ph.split.us.split:                   ; preds = %for.body3.lr.ph.split.us.for.body3.lr.ph.split.us.split_crit_edge
  br label %for.body3.us

for.body3.us:                                     ; preds = %for.cond2.loopexit.us, %for.body3.lr.ph.split.us.split
  br i1 true, label %cond.false.us, label %cond.end.us

cond.false.us:                                    ; preds = %for.body3.us
  br label %cond.end.us

cond.end.us:                                      ; preds = %cond.false.us, %for.body3.us
  %cond.us = phi i32 [ %div, %cond.false.us ], [ %conv7, %for.body3.us ]
  %6 = load i32* @b, align 4
  %cmp91.us = icmp slt i32 %6, 1
  br i1 %cmp91.us, label %for.inc.lr.ph.us, label %for.cond2.loopexit.us

for.inc.us:                                       ; preds = %for.inc.lr.ph.us, %for.inc.us
  %7 = phi i32 [ %6, %for.inc.lr.ph.us ], [ %inc.us, %for.inc.us ]
  %inc.us = add nsw i32 %7, 1
  %cmp9.us = icmp slt i32 %inc.us, 1
  br i1 %cmp9.us, label %for.inc.us, label %for.cond8.for.cond2.loopexit_crit_edge.us

for.cond2.loopexit.us:                            ; preds = %for.cond8.for.cond2.loopexit_crit_edge.us, %cond.end.us
  br i1 false, label %for.cond2.for.inc13_crit_edge.us-lcssa.us.us-lcssa, label %for.body3.us

for.inc.lr.ph.us:                                 ; preds = %cond.end.us
  br label %for.inc.us

for.cond8.for.cond2.loopexit_crit_edge.us:        ; preds = %for.inc.us
  %inc.lcssa.us = phi i32 [ %inc.us, %for.inc.us ]
  store i32 %inc.lcssa.us, i32* @b, align 4
  br label %for.cond2.loopexit.us

for.cond2.for.inc13_crit_edge.us-lcssa.us.us-lcssa: ; preds = %for.cond2.loopexit.us
  %cond.lcssa.ph.us.ph = phi i32 [ %cond.us, %for.cond2.loopexit.us ]
  br label %for.cond2.for.inc13_crit_edge.us-lcssa.us

for.cond2.for.inc13_crit_edge.us-lcssa.us:        ; preds = %for.cond2.for.inc13_crit_edge.us-lcssa.us.us-lcssa, %for.cond2.for.inc13_crit_edge.us-lcssa.us.us-lcssa.us
  %cond.lcssa.ph.us = phi i32 [ %cond.lcssa.ph.us.ph, %for.cond2.for.inc13_crit_edge.us-lcssa.us.us-lcssa ], [ %cond.lcssa.ph.us.ph.us, %for.cond2.for.inc13_crit_edge.us-lcssa.us.us-lcssa.us ]
  br label %for.cond2.for.inc13_crit_edge

for.body3.lr.ph.split:                            ; preds = %for.body3.lr.ph.for.body3.lr.ph.split_crit_edge
  br i1 %tobool, label %for.body3.lr.ph.split.split.us, label %for.body3.lr.ph.split.for.body3.lr.ph.split.split_crit_edge

for.body3.lr.ph.split.for.body3.lr.ph.split.split_crit_edge: ; preds = %for.body3.lr.ph.split
  br label %for.body3.lr.ph.split.split

for.body3.lr.ph.split.split.us:                   ; preds = %for.body3.lr.ph.split
  br label %for.body3.us3

for.body3.us3:                                    ; preds = %for.cond2.loopexit.us11, %for.body3.lr.ph.split.split.us
  br i1 false, label %cond.false.us4, label %cond.end.us5

cond.false.us4:                                   ; preds = %for.body3.us3
  br label %cond.end.us5

cond.end.us5:                                     ; preds = %cond.false.us4, %for.body3.us3
  %cond.us6 = phi i32 [ %div, %cond.false.us4 ], [ %conv7, %for.body3.us3 ]
  %8 = load i32* @b, align 4
  %cmp91.us7 = icmp slt i32 %8, 1
  br i1 %cmp91.us7, label %for.inc.lr.ph.us12, label %for.cond2.loopexit.us11

for.inc.us8:                                      ; preds = %for.inc.lr.ph.us12, %for.inc.us8
  %9 = phi i32 [ %8, %for.inc.lr.ph.us12 ], [ %inc.us9, %for.inc.us8 ]
  %inc.us9 = add nsw i32 %9, 1
  %cmp9.us10 = icmp slt i32 %inc.us9, 1
  br i1 %cmp9.us10, label %for.inc.us8, label %for.cond8.for.cond2.loopexit_crit_edge.us13

for.cond2.loopexit.us11:                          ; preds = %for.cond8.for.cond2.loopexit_crit_edge.us13, %cond.end.us5
  br i1 true, label %for.cond2.for.inc13_crit_edge.us-lcssa.us-lcssa.us, label %for.body3.us3

for.inc.lr.ph.us12:                               ; preds = %cond.end.us5
  br label %for.inc.us8

for.cond8.for.cond2.loopexit_crit_edge.us13:      ; preds = %for.inc.us8
  %inc.lcssa.us14 = phi i32 [ %inc.us9, %for.inc.us8 ]
  store i32 %inc.lcssa.us14, i32* @b, align 4
  br label %for.cond2.loopexit.us11

for.cond2.for.inc13_crit_edge.us-lcssa.us-lcssa.us: ; preds = %for.cond2.loopexit.us11
  %cond.lcssa.ph.ph.us = phi i32 [ %cond.us6, %for.cond2.loopexit.us11 ]
  br label %for.cond2.for.inc13_crit_edge.us-lcssa

for.body3.lr.ph.split.split:                      ; preds = %for.body3.lr.ph.split.for.body3.lr.ph.split.split_crit_edge
  br label %for.body3

for.cond8.for.cond2.loopexit_crit_edge:           ; preds = %for.inc
  %inc.lcssa = phi i32 [ %inc, %for.inc ]
  store i32 %inc.lcssa, i32* @b, align 4
  br label %for.cond2.loopexit

for.cond2.loopexit:                               ; preds = %cond.end, %for.cond8.for.cond2.loopexit_crit_edge
  br i1 false, label %for.cond2.for.inc13_crit_edge.us-lcssa.us-lcssa, label %for.body3

for.body3:                                        ; preds = %for.cond2.loopexit, %for.body3.lr.ph.split.split
  br i1 false, label %cond.false, label %cond.end

cond.false:                                       ; preds = %for.body3
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %for.body3
  %cond = phi i32 [ %div, %cond.false ], [ %conv7, %for.body3 ]
  %10 = load i32* @b, align 4
  %cmp91 = icmp slt i32 %10, 1
  br i1 %cmp91, label %for.inc.lr.ph, label %for.cond2.loopexit

for.inc.lr.ph:                                    ; preds = %cond.end
  br label %for.inc

for.inc:                                          ; preds = %for.inc, %for.inc.lr.ph
  %11 = phi i32 [ %10, %for.inc.lr.ph ], [ %inc, %for.inc ]
  %inc = add nsw i32 %11, 1
  %cmp9 = icmp slt i32 %inc, 1
  br i1 %cmp9, label %for.inc, label %for.cond8.for.cond2.loopexit_crit_edge

for.cond2.for.inc13_crit_edge.us-lcssa.us-lcssa:  ; preds = %for.cond2.loopexit
  %cond.lcssa.ph.ph = phi i32 [ %cond, %for.cond2.loopexit ]
  br label %for.cond2.for.inc13_crit_edge.us-lcssa

for.cond2.for.inc13_crit_edge.us-lcssa:           ; preds = %for.cond2.for.inc13_crit_edge.us-lcssa.us-lcssa, %for.cond2.for.inc13_crit_edge.us-lcssa.us-lcssa.us
  %cond.lcssa.ph = phi i32 [ %cond.lcssa.ph.ph, %for.cond2.for.inc13_crit_edge.us-lcssa.us-lcssa ], [ %cond.lcssa.ph.ph.us, %for.cond2.for.inc13_crit_edge.us-lcssa.us-lcssa.us ]
  br label %for.cond2.for.inc13_crit_edge

for.cond2.for.inc13_crit_edge:                    ; preds = %for.cond2.for.inc13_crit_edge.us-lcssa, %for.cond2.for.inc13_crit_edge.us-lcssa.us
  %cond.lcssa = phi i32 [ %cond.lcssa.ph, %for.cond2.for.inc13_crit_edge.us-lcssa ], [ %cond.lcssa.ph.us, %for.cond2.for.inc13_crit_edge.us-lcssa.us ]
  store i32 %cond.lcssa, i32* @c, align 4
  br label %for.inc13

; CHECK: [[for_inc13]]:
; CHECK-NEXT: %[[indvars_iv_next]] = add nuw nsw i32 %[[indvars_iv]], 1
; CHECK-NEXT: %[[exitcond4:.*]] = icmp ne i32 %[[indvars_iv]], -1
; CHECK-NEXT: br i1 %[[exitcond4]], label %[[for_cond2_preheader]], label %[[for_end15:.*]]
for.inc13:                                        ; preds = %for.cond2.for.inc13_crit_edge, %for.cond2.preheader
  %inc14 = add i8 %storemerge15, 1
  %cmp = icmp ugt i8 %inc14, 50
  br i1 %cmp, label %for.cond2.preheader, label %for.end15

; CHECK: [[for_end15]]:
; CHECK-NEXT: ret void
for.end15:                                        ; preds = %for.inc13
  ret void
}
