; RUN: opt < %s -mtriple=x86_64-unknown-unknown -passes=mergeicmps -verify-dom-info -S | FileCheck %s

; No adjacent accesses to the same pointer so nothing should be merged. Select blocks won't get split.

; CHECK: [[MEMCMP_OP:@memcmp_const_op]] = private constant <{ i8, i8 }> <{ i8 1, i8 9 }>

define dso_local noundef zeroext i1 @unmergable_select(
    ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %p) local_unnamed_addr {
; CHECK-LABEL: @unmergable_select(
; CHECK:       entry:
; CHECK-NEXT:    [[IDX0:%.*]] = getelementptr inbounds nuw i8, ptr [[P:%.*]], i64 10
; CHECK-NEXT:    [[TMP0:%.*]] = load i8, ptr [[IDX0]], align 1
; CHECK-NEXT:    [[IDX1:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 1
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[IDX1]], align 1
; CHECK-NEXT:    [[IDX2:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 3
; CHECK-NEXT:    [[TMP2:%.*]] = load i8, ptr [[IDX2]], align 1
; CHECK-NEXT:    [[CMP0:%.*]] = icmp eq i8 [[TMP0]], -1
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[TMP1]], -56
; CHECK-NEXT:    [[SEL0:%.*]] = select i1 [[CMP0]], i1 [[CMP1]], i1 false
; CHECK-NEXT:    [[CMP2:%.*]] = icmp eq i8 [[TMP2]], -66
; CHECK-NEXT:    [[SEL1:%.*]] = select i1 [[SEL0]], i1 [[CMP2]], i1 false
; CHECK-NEXT:    br i1 [[SEL1]], label [[LAND_LHS_11:%.*]], label [[LAND_END:%.*]]
; CHECK:       land.lhs.true11:
; CHECK-NEXT:    [[IDX3:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 12
; CHECK-NEXT:    [[TMP3:%.*]] = load i8, ptr [[IDX3]], align 1
; CHECK-NEXT:    [[CMP3:%.*]] = icmp eq i8 [[TMP3]], 1
; CHECK-NEXT:    br i1 [[CMP3]], label [[LAND_LHS_16:%.*]], label [[LAND_END]]
; CHECK:       land.lhs.true16:
; CHECK-NEXT:    [[IDX4:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 6
; CHECK-NEXT:    [[TMP4:%.*]] = load i8, ptr [[IDX4]], align 1
; CHECK-NEXT:    [[CMP4:%.*]] = icmp eq i8 [[TMP4]], 2
; CHECK-NEXT:    br i1 [[CMP4]], label [[LAND_LHS_21:%.*]], label [[LAND_END]]
; CHECK:       land.lhs.true21:
; CHECK-NEXT:    [[IDX5:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 8
; CHECK-NEXT:    [[TMP5:%.*]] = load i8, ptr [[IDX5]], align 1
; CHECK-NEXT:    [[CMP5:%.*]] = icmp eq i8 [[TMP5]], 7
; CHECK-NEXT:    br i1 [[CMP5]], label [[LAND_RHS:%.*]], label [[LAND_END]]
; CHECK:       land.rhs:
; CHECK-NEXT:    [[IDX6:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 14
; CHECK-NEXT:    [[TMP6:%.*]] = load i8, ptr [[IDX6]], align 1
; CHECK-NEXT:    [[CMP6:%.*]] = icmp eq i8 [[TMP6]], 9
; CHECK-NEXT:    br label [[LAND_END]]
; CHECK:  land.end:
; CHECK-NEXT:    [[RES:%.*]] = phi i1 [ false, [[LAND_LHS_21]] ], [ false, [[LAND_LHS_16]] ], [ false, [[LAND_LHS_11]] ], [ false, %entry ], [ %cmp28, [[LAND_RHS]] ]
; CHECK-NEXT:    ret i1 [[RES]]
;
entry:
  %arrayidx = getelementptr inbounds nuw i8, ptr %p, i64 10
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %p, i64 1
  %1 = load i8, ptr %arrayidx1, align 1
  %arrayidx2 = getelementptr inbounds nuw i8, ptr %p, i64 3
  %2 = load i8, ptr %arrayidx2, align 1
  %cmp = icmp eq i8 %0, -1
  %cmp5 = icmp eq i8 %1, -56
  %or.cond = select i1 %cmp, i1 %cmp5, i1 false
  %cmp9 = icmp eq i8 %2, -66
  %or.cond30 = select i1 %or.cond, i1 %cmp9, i1 false
  br i1 %or.cond30, label %land.lhs.true11, label %land.end

land.lhs.true11:                                  ; preds = %entry
  %arrayidx12 = getelementptr inbounds nuw i8, ptr %p, i64 12
  %3 = load i8, ptr %arrayidx12, align 1
  %cmp14 = icmp eq i8 %3, 1
  br i1 %cmp14, label %land.lhs.true16, label %land.end

land.lhs.true16:                                  ; preds = %land.lhs.true11
  %arrayidx17 = getelementptr inbounds nuw i8, ptr %p, i64 6
  %4 = load i8, ptr %arrayidx17, align 1
  %cmp19 = icmp eq i8 %4, 2
  br i1 %cmp19, label %land.lhs.true21, label %land.end

land.lhs.true21:                                  ; preds = %land.lhs.true16
  %arrayidx22 = getelementptr inbounds nuw i8, ptr %p, i64 8
  %5 = load i8, ptr %arrayidx22, align 1
  %cmp24 = icmp eq i8 %5, 7
  br i1 %cmp24, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %land.lhs.true21
  %arrayidx26 = getelementptr inbounds nuw i8, ptr %p, i64 14
  %6 = load i8, ptr %arrayidx26, align 1
  %cmp28 = icmp eq i8 %6, 9
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true21, %land.lhs.true16, %land.lhs.true11, %entry
  %7 = phi i1 [ false, %land.lhs.true21 ], [ false, %land.lhs.true16 ], [ false, %land.lhs.true11 ], [ false, %entry ], [ %cmp28, %land.rhs ]
  ret i1 %7
}

; p[12] and p[13] mergable, select mult-block is part of the chain but isn't merged and won't get split up into its single comparisons.

define dso_local noundef zeroext i1 @partial_merge_not_select(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %p) local_unnamed_addr {
; CHECK-LABEL: @partial_merge_not_select(
; CHECK:       entry3:
; CHECK-NEXT:    [[IDX0:%.*]] = getelementptr inbounds nuw i8, ptr [[P:%.*]], i64 10
; CHECK-NEXT:    [[TMP0:%.*]] = load i8, ptr [[IDX0]], align 1
; CHECK-NEXT:    [[IDX1:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 1
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[IDX1]], align 1
; CHECK-NEXT:    [[IDX2:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 3
; CHECK-NEXT:    [[TMP2:%.*]] = load i8, ptr [[IDX2]], align 1
; CHECK-NEXT:    [[CMP0:%.*]] = icmp eq i8 [[TMP0]], -1
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i8 [[TMP1]], -56
; CHECK-NEXT:    [[SEL0:%.*]] = select i1 [[CMP0]], i1 [[CMP1]], i1 false
; CHECK-NEXT:    [[CMP2:%.*]] = icmp eq i8 [[TMP2]], -66
; CHECK-NEXT:    [[SEL1:%.*]] = select i1 [[SEL0]], i1 [[CMP2]], i1 false
; CHECK-NEXT:    br i1 [[SEL1]], label [[LAND_LHS_LAND_RHS:%.*]], label [[LAND_END:%.*]]
; CHECK:       "land.lhs.true11+land.rhs":
; CHECK-NEXT:    [[IDX3:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 12
; CHECK-NEXT:    [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[IDX3]], ptr [[MEMCMP_OP]], i64 2)
; CHECK-NEXT:    [[CMP3:%.*]] = icmp eq i32 [[MEMCMP]], 0
; CHECK-NEXT:    br i1 [[CMP3]], label [[LAND_LHS_16:%.*]], label [[LAND_END]]
; CHECK:       land.lhs.true162:
; CHECK-NEXT:    [[IDX4:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 6
; CHECK-NEXT:    [[TMP4:%.*]] = load i8, ptr [[IDX4]], align 1
; CHECK-NEXT:    [[CMP4:%.*]] = icmp eq i8 [[TMP4]], 2
; CHECK-NEXT:    br i1 [[CMP4]], label [[LAND_LHS_21:%.*]], label [[LAND_END]]
; CHECK:       land.lhs.true211:
; CHECK-NEXT:    [[IDX5:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 8
; CHECK-NEXT:    [[TMP5:%.*]] = load i8, ptr [[IDX5]], align 1
; CHECK-NEXT:    [[CMP5:%.*]] = icmp eq i8 [[TMP5]], 7
; CHECK-NEXT:    br label [[LAND_END]]
; CHECK:  land.end:
; CHECK-NEXT:    [[RES:%.*]] = phi i1 [ [[CMP5]], [[LAND_LHS_21]] ], [ false, [[LAND_LHS_16]] ], [ false, [[LAND_LHS_LAND_RHS]] ], [ false, %entry3 ]
; CHECK-NEXT:    ret i1 [[RES]]
;
entry:
  %arrayidx = getelementptr inbounds nuw i8, ptr %p, i64 10
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %p, i64 1
  %1 = load i8, ptr %arrayidx1, align 1
  %arrayidx2 = getelementptr inbounds nuw i8, ptr %p, i64 3
  %2 = load i8, ptr %arrayidx2, align 1
  %cmp = icmp eq i8 %0, -1
  %cmp5 = icmp eq i8 %1, -56
  %or.cond = select i1 %cmp, i1 %cmp5, i1 false
  %cmp9 = icmp eq i8 %2, -66
  %or.cond30 = select i1 %or.cond, i1 %cmp9, i1 false
  br i1 %or.cond30, label %land.lhs.true11, label %land.end

land.lhs.true11:                                  ; preds = %entry
  %arrayidx12 = getelementptr inbounds nuw i8, ptr %p, i64 12
  %3 = load i8, ptr %arrayidx12, align 1
  %cmp14 = icmp eq i8 %3, 1
  br i1 %cmp14, label %land.lhs.true16, label %land.end

land.lhs.true16:                                  ; preds = %land.lhs.true11
  %arrayidx17 = getelementptr inbounds nuw i8, ptr %p, i64 6
  %4 = load i8, ptr %arrayidx17, align 1
  %cmp19 = icmp eq i8 %4, 2
  br i1 %cmp19, label %land.lhs.true21, label %land.end

land.lhs.true21:                                  ; preds = %land.lhs.true16
  %arrayidx22 = getelementptr inbounds nuw i8, ptr %p, i64 8
  %5 = load i8, ptr %arrayidx22, align 1
  %cmp24 = icmp eq i8 %5, 7
  br i1 %cmp24, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %land.lhs.true21
  %arrayidx26 = getelementptr inbounds nuw i8, ptr %p, i64 13
  %6 = load i8, ptr %arrayidx26, align 1
  %cmp28 = icmp eq i8 %6, 9
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true21, %land.lhs.true16, %land.lhs.true11, %entry
  %7 = phi i1 [ false, %land.lhs.true21 ], [ false, %land.lhs.true16 ], [ false, %land.lhs.true11 ], [ false, %entry ], [ %cmp28, %land.rhs ]
  ret i1 %7
}
