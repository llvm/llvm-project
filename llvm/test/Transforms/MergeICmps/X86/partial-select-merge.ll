; RUN: opt < %s -mtriple=x86_64-unknown-unknown -passes=mergeicmps -verify-dom-info -S | FileCheck %s --check-prefix=REG

; Cannot merge only part of a select block if not entire block mergable.

define zeroext i1 @cmp_partially_mergable_select(
    ptr nocapture readonly align 4 dereferenceable(24) %a,
    ptr nocapture readonly align 4 dereferenceable(24) %b) local_unnamed_addr {
; REG-LABEL: @cmp_partially_mergable_select(
; REG:      entry:
; REG-NEXT:   [[IDX0:%.*]] = getelementptr inbounds nuw i8, ptr [[A:%.*]], i64 8
; REG-NEXT:   [[TMP0:%.*]] = load i32, ptr [[IDX0]], align 4
; REG-NEXT:   [[CMP0:%.*]] = icmp eq i32 [[TMP0]], 255
; REG-NEXT:   br i1 [[CMP0]], label [[LAND_LHS_TRUE:%.*]], label [[LAND_END:%.*]]
; REG:      land.lhs.true:
; REG-NEXT:   [[TMP1:%.*]] = load i32, ptr [[A]], align 4
; REG-NEXT:   [[TMP2:%.*]] = load i32, ptr [[B:%.*]], align 4
; REG-NEXT:   [[CMP1:%.*]] = icmp eq i32 [[TMP1]], [[TMP2]]
; REG-NEXT:   br i1 [[CMP1]], label [[LAND_LHS_TRUE_4:%.*]], label [[LAND_END]]
; REG:      land.lhs.true4:
; REG-NEXT:   [[IDX1:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 5
; REG-NEXT:   [[TMP3:%.*]] = load i8, ptr [[IDX1]], align 1
; REG-NEXT:   [[IDX2:%.*]] = getelementptr inbounds nuw i8, ptr [[B]], i64 5
; REG-NEXT:   [[TMP4:%.*]] = load i8, ptr [[IDX2]], align 1
; REG-NEXT:   [[CMP2:%.*]] = icmp eq i8 [[TMP3]], [[TMP4]]
; REG-NEXT:   [[IDX3:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 16
; REG-NEXT:   [[TMP5:%.*]] = load i32, ptr [[IDX3]], align 4
; REG-NEXT:   [[CMP3:%.*]] = icmp eq i32 [[TMP5]], 100
; REG-NEXT:   [[SEL0:%.*]] = select i1 [[CMP2]], i1 [[CMP3]], i1 false
; REG-NEXT:   br i1 [[SEL0]], label [[LAND_LHS_TRUE_10:%.*]], label [[LAND_END]]
; REG:      land.lhs.true10:
; REG-NEXT:   [[IDX4:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 20
; REG-NEXT:   [[TMP6:%.*]] = load i8, ptr [[IDX4]], align 4
; REG-NEXT:   [[IDX5:%.*]] = getelementptr inbounds nuw i8, ptr [[B]], i64 20
; REG-NEXT:   [[TMP7:%.*]] = load i8, ptr [[IDX5]], align 4
; REG-NEXT:   [[CMP4:%.*]] = icmp eq i8 [[TMP6]], [[TMP7]]
; REG-NEXT:   br i1 [[CMP4]], label [[LAND_RHS:%.*]], label [[LAND_END]]
; REG:      land.rhs:
; REG-NEXT:   [[IDX6:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 4
; REG-NEXT:   [[TMP8:%.*]] = load i8, ptr [[IDX6]], align 4
; REG-NEXT:   [[IDX7:%.*]] = getelementptr inbounds nuw i8, ptr [[B]], i64 4
; REG-NEXT:   [[TMP9:%.*]] = load i8, ptr [[IDX7]], align 4
; REG-NEXT:   [[CMP5:%.*]] = icmp eq i8 [[TMP8]], [[TMP9]]
; REG-NEXT:   br label [[LAND_END]]
; REG:      land.end:
; REG-NEXT:   [[RES:%.*]] = phi i1 [ false, [[LAND_LHS_TRUE_10]] ], [ false, [[LAND_LHS_TRUE_4]] ], [ false, [[LAND_LHS_TRUE]] ], [ false, %entry ], [ [[CMP5]], [[LAND_RHS]] ]
; REG-NEXT:   ret i1 [[RES]]
;
entry:
  %e = getelementptr inbounds nuw i8, ptr %a, i64 8
  %0 = load i32, ptr %e, align 4
  %cmp = icmp eq i32 %0, 255
  br i1 %cmp, label %land.lhs.true, label %land.end

land.lhs.true:                                    ; preds = %entry
  %1 = load i32, ptr %a, align 4
  %2 = load i32, ptr %b, align 4
  %cmp3 = icmp eq i32 %1, %2
  br i1 %cmp3, label %land.lhs.true4, label %land.end

land.lhs.true4:                                   ; preds = %land.lhs.true
  %c = getelementptr inbounds nuw i8, ptr %a, i64 5
  %3 = load i8, ptr %c, align 1
  %c5 = getelementptr inbounds nuw i8, ptr %b, i64 5
  %4 = load i8, ptr %c5, align 1
  %cmp7 = icmp eq i8 %3, %4
  %g = getelementptr inbounds nuw i8, ptr %a, i64 16
  %5 = load i32, ptr %g, align 4
  %cmp9 = icmp eq i32 %5, 100
  %or.cond = select i1 %cmp7, i1 %cmp9, i1 false
  br i1 %or.cond, label %land.lhs.true10, label %land.end

land.lhs.true10:                                  ; preds = %land.lhs.true4
  %h = getelementptr inbounds nuw i8, ptr %a, i64 20
  %6 = load i8, ptr %h, align 4
  %h12 = getelementptr inbounds nuw i8, ptr %b, i64 20
  %7 = load i8, ptr %h12, align 4
  %cmp14 = icmp eq i8 %6, %7
  br i1 %cmp14, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %land.lhs.true10
  %b15 = getelementptr inbounds nuw i8, ptr %a, i64 4
  %8 = load i8, ptr %b15, align 4
  %b17 = getelementptr inbounds nuw i8, ptr %b, i64 4
  %9 = load i8, ptr %b17, align 4
  %cmp19 = icmp eq i8 %8, %9
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true10, %land.lhs.true4, %land.lhs.true, %entry
  %10 = phi i1 [ false, %land.lhs.true10 ], [ false, %land.lhs.true4 ], [ false, %land.lhs.true ], [ false, %entry ], [ %cmp19, %land.rhs ]
  ret i1 %10
}


; p[12] and p[13] are mergable. p[12] is inside of a select block which will not be split up, so it shouldn't merge them.

define dso_local zeroext i1 @cmp_partially_mergable_select_array(
    ptr nocapture readonly align 1 dereferenceable(24) %p) local_unnamed_addr {
; REG-LABEL: @cmp_partially_mergable_select_array(
; REG:       entry:
; REG-NEXT:   [[IDX0:%.*]] = getelementptr inbounds nuw i8, ptr [[P:%.*]], i64 12
; REG-NEXT:   [[TMP0:%.*]] = load i8, ptr [[IDX0]], align 1
; REG-NEXT:   [[IDX1:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 1
; REG-NEXT:   [[TMP1:%.*]] = load i8, ptr [[IDX1]], align 1
; REG-NEXT:   [[IDX2:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 3
; REG-NEXT:   [[TMP2:%.*]] = load i8, ptr [[IDX2]], align 1
; REG-NEXT:   [[CMP0:%.*]] = icmp eq i8 [[TMP0]], -1
; REG-NEXT:   [[CMP1:%.*]] = icmp eq i8 [[TMP1]], -56
; REG-NEXT:   [[SEL0:%.*]] = select i1 [[CMP0]], i1 [[CMP1]], i1 false
; REG-NEXT:   [[CMP2:%.*]] = icmp eq i8 [[TMP2]], -66
; REG-NEXT:   [[SEL1:%.*]] = select i1 [[SEL0]], i1 [[CMP2]], i1 false
; REG-NEXT:   br i1 [[SEL1]], label [[LAND_LHS_TRUE_11:%.*]], label [[LAND_END:%.*]]
; REG:       land.lhs.true11:
; REG-NEXT:   [[IDX3:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 10
; REG-NEXT:   [[TMP3:%.*]] = load i8, ptr [[IDX3]], align 1
; REG-NEXT:   [[CMP3:%.*]] = icmp eq i8 [[TMP3]], 1
; REG-NEXT:   br i1 [[CMP3]], label [[LAND_LHS_TRUE_16:%.*]], label [[LAND_END]]
; REG:       land.lhs.true16:
; REG-NEXT:   [[IDX4:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 6
; REG-NEXT:   [[TMP4:%.*]] = load i8, ptr [[IDX4]], align 1
; REG-NEXT:   [[CMP4:%.*]] = icmp eq i8 [[TMP4]], 2
; REG-NEXT:   br i1 [[CMP4]], label [[LAND_LHS_TRUE_21:%.*]], label [[LAND_END]]
; REG:       land.lhs.true21:
; REG-NEXT:   [[IDX5:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 8
; REG-NEXT:   [[TMP5:%.*]] = load i8, ptr [[IDX5]], align 1
; REG-NEXT:   [[CMP5:%.*]] = icmp eq i8 [[TMP5]], 7
; REG-NEXT:   br i1 [[CMP5]], label [[LAND_RHS:%.*]], label [[LAND_END]]
; REG:       land.rhs:
; REG-NEXT:   [[IDX6:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 13
; REG-NEXT:   [[TMP6:%.*]] = load i8, ptr [[IDX6]], align 1
; REG-NEXT:   [[CMP6:%.*]] = icmp eq i8 [[TMP6]], 9
; REG-NEXT:   br label [[LAND_END]]
; REG:       land.end:
; REG-NEXT:   [[RES:%.*]] = phi i1 [ false, [[LAND_LHS_TRUE_21]] ], [ false, [[LAND_LHS_TRUE_16]] ], [ false, [[LAND_LHS_TRUE_11]] ], [ false, %entry ], [ [[CMP6]], [[LAND_RHS]] ]
; REG-NEXT:   ret i1 [[RES]]
;
entry:
  %arrayidx = getelementptr inbounds nuw i8, ptr %p, i64 12
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

land.lhs.true11:
  %arrayidx12 = getelementptr inbounds nuw i8, ptr %p, i64 10
  %3 = load i8, ptr %arrayidx12, align 1
  %cmp14 = icmp eq i8 %3, 1
  br i1 %cmp14, label %land.lhs.true16, label %land.end

land.lhs.true16:
  %arrayidx17 = getelementptr inbounds nuw i8, ptr %p, i64 6
  %4 = load i8, ptr %arrayidx17, align 1
  %cmp19 = icmp eq i8 %4, 2
  br i1 %cmp19, label %land.lhs.true21, label %land.end

land.lhs.true21:
  %arrayidx22 = getelementptr inbounds nuw i8, ptr %p, i64 8
  %5 = load i8, ptr %arrayidx22, align 1
  %cmp24 = icmp eq i8 %5, 7
  br i1 %cmp24, label %land.rhs, label %land.end

land.rhs:
  %arrayidx26 = getelementptr inbounds nuw i8, ptr %p, i64 13
  %6 = load i8, ptr %arrayidx26, align 1
  %cmp28 = icmp eq i8 %6, 9
  br label %land.end

land.end:
  %7 = phi i1 [ false, %land.lhs.true21 ], [ false, %land.lhs.true16 ], [ false, %land.lhs.true11 ], [ false, %entry ], [ %cmp28, %land.rhs ]
  ret i1 %7
}

