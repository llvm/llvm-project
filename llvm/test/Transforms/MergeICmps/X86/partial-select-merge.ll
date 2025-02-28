; RUN: opt < %s -mtriple=x86_64-unknown-unknown -passes=mergeicmps -verify-dom-info -S | FileCheck %s --check-prefix=REG
; RUN: opt < %s -mtriple=x86_64-unknown-unknown -passes='mergeicmps,simplifycfg' -verify-dom-info -S | FileCheck %s --check-prefix=CFG

; REG checks the IR when only mergeicmps is run.
; CFG checks the IR when simplifycfg is run afterwards to merge distinct blocks back together.

; Can merge part of a select block even if not entire block mergable.

define zeroext i1 @cmp_partially_mergable_select(
    ptr nocapture readonly align 4 dereferenceable(24) %a,
    ptr nocapture readonly align 4 dereferenceable(24) %b) local_unnamed_addr {
; REG-LABEL: @cmp_partially_mergable_select(
; REG:      "land.lhs.true+land.rhs+land.lhs.true4":
; REG-NEXT:   [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[A:%.*]], ptr [[B:%.*]], i64 6)
; REG-NEXT:   [[CMP1:%.*]] = icmp eq i32 [[MEMCMP]], 0
; REG-NEXT:   br i1 [[CMP1]], label [[LAND_LHS_103:%.*]], label [[LAND_END:%.*]]
; REG:      land.lhs.true103:
; REG-NEXT:   [[TMP0:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 20
; REG-NEXT:   [[TMP1:%.*]] = getelementptr inbounds nuw i8, ptr [[B]], i64 20
; REG-NEXT:   [[TMP2:%.*]] = load i8, ptr [[TMP0]], align 4
; REG-NEXT:   [[TMP3:%.*]] = load i8, ptr [[TMP1]], align 4
; REG-NEXT:   [[CMP2:%.*]] = icmp eq i8 [[TMP2]], [[TMP3]]
; REG-NEXT:   br i1 [[CMP2]], label [[ENTRY2:%.*]], label [[LAND_END]]
; REG:      entry2:
; REG-NEXT:   [[TMP4:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 8
; REG-NEXT:   [[TMP5:%.*]] = load i32, ptr [[TMP4]], align 4
; REG-NEXT:   [[CMP3:%.*]] = icmp eq i32 [[TMP5]], 255
; REG-NEXT:   br i1 [[CMP3]], label [[LAND_LHS_41:%.*]], label [[LAND_END]]
; REG:      land.lhs.true41:
; REG-NEXT:   [[TMP6:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 16
; REG-NEXT:   [[TMP7:%.*]] = load i32, ptr [[TMP6]], align 4
; REG-NEXT:   [[CMP4:%.*]] = icmp eq i32 [[TMP7]], 100
; REG-NEXT:   br label %land.end
; REG:      land.end:
; REG-NEXT:   [[TMP8:%.*]] = phi i1 [ [[CMP4]], [[LAND_LHS_41]] ], [ false, [[ENTRY2]] ], [ false, [[LAND_LHS_103]] ], [ false, %"land.lhs.true+land.rhs+land.lhs.true4" ]
; REG-NEXT:   ret i1 [[TMP8]]
;
; CFG-LABEL: @cmp_partially_mergable_select(
; CFG:      "land.lhs.true+land.rhs+land.lhs.true4":
; CFG-NEXT:   [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[A:%.*]], ptr [[B:%.*]], i64 6)
; CFG-NEXT:   [[CMP1:%.*]] = icmp eq i32 [[MEMCMP]], 0
; CFG-NEXT:   br i1 [[CMP1]], label [[LAND_LHS_103:%.*]], label [[LAND_END:%.*]]
; CFG:      land.lhs.true103:
; CFG-NEXT:   [[TMP0:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 20
; CFG-NEXT:   [[TMP1:%.*]] = getelementptr inbounds nuw i8, ptr [[B]], i64 20
; CFG-NEXT:   [[TMP2:%.*]] = load i8, ptr [[TMP0]], align 4
; CFG-NEXT:   [[TMP3:%.*]] = load i8, ptr [[TMP1]], align 4
; CFG-NEXT:   [[CMP2:%.*]] = icmp eq i8 [[TMP2]], [[TMP3]]
; CFG-NEXT:   [[TMP4:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 8
; CFG-NEXT:   [[TMP5:%.*]] = load i32, ptr [[TMP4]], align 4
; CFG-NEXT:   [[CMP3:%.*]] = icmp eq i32 [[TMP5]], 255
; CFG-NEXT:   [[SEL:%.*]] = select i1 %5, i1 %8, i1 false
; CFG-NEXT:   br i1 [[SEL]], label [[LAND_LHS_41:%.*]], label [[LAND_END]]
; CFG:      land.lhs.true41:
; CFG-NEXT:   [[TMP6:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 16
; CFG-NEXT:   [[TMP7:%.*]] = load i32, ptr [[TMP6]], align 4
; CFG-NEXT:   [[CMP4:%.*]] = icmp eq i32 [[TMP7]], 100
; CFG-NEXT:   br label %land.end
; CFG:      land.end:
; CFG-NEXT:   [[RES:%.*]] = phi i1 [ [[CMP4]], [[LAND_LHS_41]] ], [ false, [[LAND_LHS_103]] ], [ false, %"land.lhs.true+land.rhs+land.lhs.true4" ]
; CFG-NEXT:   ret i1 [[RES]]
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


; p[12] and p[13] are mergable. p[12] is inside of a select block which will be split up.
; MergeICmps always splits up matching select blocks. The following simplifycfg pass merges them back together.

define dso_local zeroext i1 @cmp_partially_mergable_select_array(
    ptr nocapture readonly align 1 dereferenceable(24) %p) local_unnamed_addr {
; REG-LABEL: @cmp_partially_mergable_select_array(
; REG: "entry+land.rhs":
; REG-NEXT:   [[IDX0:%.*]] = getelementptr inbounds nuw i8, ptr [[P:%.*]], i64 12
; REG-NEXT:   [[TMP0:%.*]] = alloca [2 x i8], align 1
; REG-NEXT:   store [2 x i8] c"\FF\09", ptr [[TMP0]], align 1
; REG-NEXT:   [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[IDX0]], ptr [[TMP0]], i64 2)
; REG-NEXT:   [[CMP0:%.*]] = icmp eq i32 [[MEMCMP]], 0
; REG-NEXT:   br i1 [[CMP0]], label [[ENTRY_5:%.*]], label [[LAND_END:%.*]]
; REG: entry5:
; REG-NEXT:   [[IDX1:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 1
; REG-NEXT:   [[TMP1:%.*]] = load i8, ptr [[IDX1]], align 1
; REG-NEXT:   [[CMP1:%.*]] = icmp eq i8 [[TMP1]], -56
; REG-NEXT:   br i1 [[CMP1]], label [[ENTRY_4:%.*]], label [[LAND_END:%.*]]
; REG: entry4:
; REG-NEXT:   [[IDX2:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 3
; REG-NEXT:   [[TMP2:%.*]] = load i8, ptr [[IDX2]], align 1
; REG-NEXT:   [[CMP2:%.*]] = icmp eq i8 [[TMP2]], -66
; REG-NEXT:   br i1 [[CMP2]], label [[LAND_LHS_113:%.*]], label [[LAND_END]]
; REG: land.lhs.true113:
; REG-NEXT:   [[IDX3:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 10
; REG-NEXT:   [[TMP3:%.*]] = load i8, ptr [[IDX3]], align 1
; REG-NEXT:   [[CMP3:%.*]] = icmp eq i8 [[TMP3]], 1
; REG-NEXT:   br i1 [[CMP3]], label [[LAND_LHS_162:%.*]], label [[LAND_END]]
; REG: land.lhs.true162:
; REG-NEXT:   [[IDX4:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 6
; REG-NEXT:   [[TMP4:%.*]] = load i8, ptr [[IDX4]], align 1
; REG-NEXT:   [[CMP4:%.*]] = icmp eq i8 [[TMP4]], 2
; REG-NEXT:   br i1 [[CMP4]], label [[LAND_LHS_211:%.*]], label [[LAND_END]]
; REG: land.lhs.true211:
; REG-NEXT:   [[IDX5:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 8
; REG-NEXT:   [[TMP5:%.*]] = load i8, ptr [[IDX5]], align 1
; REG-NEXT:   [[CMP5:%.*]] = icmp eq i8 [[TMP5]], 7
; REG-NEXT:   br label [[LAND_END]]
; REG: land.end:
; REG-NEXT:   [[RES:%.*]] = phi i1 [ [[CMP5]], [[LAND_LHS_211]] ], [ false, [[LAND_LHS_162]] ], [ false, [[LAND_LHS_113]] ], [ false, [[ENTRY_4]] ], [ false, [[ENTRY_5]] ], [ false, %"entry+land.rhs" ]
; REG-NEXT:   ret i1 [[RES]]
;
;
; CFG-LABEL: @cmp_partially_mergable_select_array(
; CFG:      "entry+land.rhs":
; CFG-NEXT:   [[IDX0:%.*]] = getelementptr inbounds nuw i8, ptr [[P:%.*]], i64 12
; CFG-NEXT:   [[TMP0:%.*]] = alloca [2 x i8], align 1
; CFG-NEXT:   store [2 x i8] c"\FF\09", ptr [[TMP0]], align 1
; CFG-NEXT:   [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[IDX0]], ptr [[TMP0]], i64 2)
; CFG-NEXT:   [[CMP0:%.*]] = icmp eq i32 [[MEMCMP]], 0
; CFG-NEXT:   [[IDX1:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 1
; CFG-NEXT:   [[TMP1:%.*]] = load i8, ptr [[IDX1]], align 1
; CFG-NEXT:   [[CMP1:%.*]] = icmp eq i8 [[TMP1:%.*]], -56
; CFG-NEXT:   [[SEL0:%.*]] = select i1 [[CMP0]], i1 [[CMP1]], i1 false
; CFG-NEXT:   [[IDX2:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 3
; CFG-NEXT:   [[TMP2:%.*]] = load i8, ptr [[IDX2]], align 1
; CFG-NEXT:   [[CMP2:%.*]] = icmp eq i8 [[TMP2]], -66
; CFG-NEXT:   [[SEL1:%.*]] = select i1 [[SEL0]], i1 [[CMP2]], i1 false
; CFG-NEXT:   [[IDX3:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 10
; CFG-NEXT:   [[TMP3:%.*]] = load i8, ptr [[IDX3]], align 1
; CFG-NEXT:   [[CMP3:%.*]] = icmp eq i8 [[TMP3]], 1
; CFG-NEXT:   [[SEL2:%.*]] = select i1 [[SEL1]], i1 [[CMP3]], i1 false
; CFG-NEXT:   [[IDX4:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 6
; CFG-NEXT:   [[TMP4:%.*]] = load i8, ptr [[IDX4]], align 1
; CFG-NEXT:   [[CMP4:%.*]] = icmp eq i8 [[TMP4]], 2
; CFG-NEXT:   [[SEL3:%.*]] = select i1 [[SEL2]], i1 [[CMP4]], i1 false
; CFG-NEXT:   br i1 [[SEL3]], label [[LAND_LHS_211:%.*]], label [[LAND_END]]
; CFG:      land.lhs.true211:
; CFG-NEXT:   [[IDX5:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 8
; CFG-NEXT:   [[TMP5:%.*]] = load i8, ptr [[IDX5]], align 1
; CFG-NEXT:   [[CMP5:%.*]] = icmp eq i8 [[TMP5]], 7
; CFG-NEXT:   br label [[LAND_END]]
; CFG:      land.end:
; CFG-NEXT:   [[RES:%.*]] = phi i1 [ [[CMP5]], [[LAND_LHS_211]] ], [ false, %"entry+land.rhs" ]
; CFG-NEXT:   ret i1 [[RES]]
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

