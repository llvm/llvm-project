; RUN: opt < %s -mtriple=x86_64-unknown-unknown -passes=mergeicmps -verify-dom-info -S 2>&1 | FileCheck %s

; Can merge contiguous const-comparison basic blocks that include a select statement.

; CHECK: [[MEMCMP_OP0:@memcmp_const_op]] = private constant <{ i8, i8 }> <{ i8 2, i8 7 }>
; CHECK: [[MEMCMP_OP1:@memcmp_const_op.1]] = private constant <{ i8, i8, i8, i8 }> <{ i8 -1, i8 -56, i8 -66, i8 1 }>

define dso_local zeroext i1 @is_all_ones_many(ptr nocapture noundef nonnull dereferenceable(24) %p) local_unnamed_addr {
; CHECK-LABEL: @is_all_ones_many(
; CHECK-NEXT:  "entry+land.lhs.true11":
; CHECK-NEXT:    [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[P:%.*]], ptr [[MEMCMP_OP1]], i64 4)
; CHECK-NEXT:    [[TMP0:%.*]] = icmp eq i32 [[MEMCMP]], 0
; CHECK-NEXT:    br i1 [[TMP0]], label [[NEXT_MEMCMP:%.*]], label [[LAND_END:%.*]]
; CHECK:  "land.lhs.true16+land.lhs.true21":
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 6
; CHECK-NEXT:    [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[TMP1]], ptr [[MEMCMP_OP0]], i64 2)
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[MEMCMP]], 0
; CHECK-NEXT:    br i1 [[TMP2]], label [[LAST_CMP:%.*]], label [[LAND_END]]
; CHECK:  land.rhs1:
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds nuw i8, ptr [[P]], i64 9
; CHECK-NEXT:    [[TMP4:%.*]] = load i8, ptr [[TMP3]], align 1
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i8 [[TMP4]], 9
; CHECK-NEXT:    br label [[LAND_END]]
; CHECK:       land.end:
; CHECK-NEXT:    [[TMP6:%.*]] = phi i1 [ [[TMP5]], [[LAST_CMP]] ], [ false, [[NEXT_MEMCMP]] ], [ false, [[ENTRY:%.*]] ]
; CHECK-NEXT:    ret i1 [[TMP6]]
;
entry:
  %0 = load i8, ptr %p, align 1
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %p, i64 1
  %1 = load i8, ptr %arrayidx1, align 1
  %arrayidx2 = getelementptr inbounds nuw i8, ptr %p, i64 2
  %2 = load i8, ptr %arrayidx2, align 1
  %cmp = icmp eq i8 %0, -1
  %cmp5 = icmp eq i8 %1, -56
  %or.cond = select i1 %cmp, i1 %cmp5, i1 false
  %cmp9 = icmp eq i8 %2, -66
  %or.cond28 = select i1 %or.cond, i1 %cmp9, i1 false
  br i1 %or.cond28, label %land.lhs.true11, label %land.end

land.lhs.true11:                                  ; preds = %entry
  %arrayidx12 = getelementptr inbounds nuw i8, ptr %p, i64 3
  %3 = load i8, ptr %arrayidx12, align 1
  %cmp14 = icmp eq i8 %3, 1
  br i1 %cmp14, label %land.lhs.true16, label %land.end

land.lhs.true16:                                  ; preds = %land.lhs.true11
  %arrayidx17 = getelementptr inbounds nuw i8, ptr %p, i64 6
  %4 = load i8, ptr %arrayidx17, align 1
  %cmp19 = icmp eq i8 %4, 2
  br i1 %cmp19, label %land.lhs.true21, label %land.end

land.lhs.true21:                                  ; preds = %land.lhs.true16
  %arrayidx22 = getelementptr inbounds nuw i8, ptr %p, i64 7
  %5 = load i8, ptr %arrayidx22, align 1
  %cmp24 = icmp eq i8 %5, 7
  br i1 %cmp24, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %land.lhs.true21
  %arrayidx26 = getelementptr inbounds nuw i8, ptr %p, i64 9
  %6 = load i8, ptr %arrayidx26, align 1
  %cmp28 = icmp eq i8 %6, 9
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true21, %land.lhs.true16, %land.lhs.true11, %entry
  %7 = phi i1 [ false, %land.lhs.true21 ], [ false, %land.lhs.true16 ], [ false, %land.lhs.true11 ], [ false, %entry ], [ %cmp28, %land.rhs ]
  ret i1 %7
}
