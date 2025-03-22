; RUN: opt < %s -mtriple=x86_64-unknown-unknown -passes=mergeicmps -verify-dom-info -S | FileCheck %s

; define dso_local noundef zeroext i1 @cmp_mixed_split(ptr noundef nonnull readonly align 4 captures(none) dereferenceable(40) %a, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(40) %b) local_unnamed_addr {
; entry:
;   %0 = load i32, ptr %a, align 4
;   %1 = load i32, ptr %b, align 4
;   %cmp = icmp eq i32 %0, %1
;   br i1 %cmp, label %land.lhs.true, label %land.end
; 
; land.lhs.true:                                    ; preds = %entry
;   %e = getelementptr inbounds nuw i8, ptr %a, i64 20
;   %2 = load i32, ptr %e, align 4
;   %a3 = getelementptr inbounds nuw i8, ptr %b, i64 4
;   %3 = load i32, ptr %a3, align 4
;   %b2 = getelementptr inbounds nuw i8, ptr %a, i64 8
;   %4 = load i32, ptr %b2, align 4
;   %c = getelementptr inbounds nuw i8, ptr %a, i64 12
;   %5 = load i8, ptr %c, align 4
;   %a1 = getelementptr inbounds nuw i8, ptr %a, i64 4
;   %6 = load i32, ptr %a1, align 4
;   %d = getelementptr inbounds nuw i8, ptr %a, i64 16
;   %7 = load i32, ptr %d, align 4
;   %cmp5 = icmp eq i32 %6, %3
;   %cmp7 = icmp eq i8 %5, 43
;   %or.cond = select i1 %cmp5, i1 %cmp7, i1 false
;   %cmp9 = icmp eq i32 %4, 1
;   %or.cond13 = select i1 %or.cond, i1 %cmp9, i1 false
;   %cmp11 = icmp eq i32 %7, 12
;   %or.cond14 = select i1 %or.cond13, i1 %cmp11, i1 false
;   %cmp12 = icmp eq i32 %2, 3
;   %spec.select = select i1 %or.cond14, i1 %cmp12, i1 false
;   br label %land.end
; 
; land.end:                                         ; preds = %land.lhs.true, %entry
;   %8 = phi i1 [ false, %entry ], [ %spec.select, %land.lhs.true ]
;   ret i1 %8
; }




declare void @foo(...)

; Tests that if both const-cmp and bce-cmp chains can be merged that the splitted block is still at the beginning.

define dso_local noundef zeroext i1 @cmp_mixed_const_first(ptr noundef nonnull align 4 dereferenceable(20) %a, ptr noundef nonnull align 4 dereferenceable(20) %b) local_unnamed_addr {
; CHECK-LABEL: @cmp_mixed_const_first(
; This merged-block should come first as it should be split.
; CHECK:  "entry+land.rhs+land.lhs.true8":
; CHECK-NEXT:    call void (...) @foo() #[[ATTR2:[0-9]+]]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw i8, ptr [[A:%.*]], i64 8
; CHECK-NEXT:    [[TMP1:%.*]] = alloca <{ i32, i32, i32 }>
; CHECK-NEXT:    store <{ i32, i32, i32 }> <{ i32 255, i32 200, i32 100 }>, ptr [[TMP1]], align 1
; CHECK-NEXT:    [[MEMCMP0:%.*]] = call i32 @memcmp(ptr [[TMP0]], ptr [[TMP1]], i64 12)
; CHECK-NEXT:    [[CMP0:%.*]] = icmp eq i32 [[MEMCMP0]], 0
; CHECK-NEXT:    br i1 [[CMP0]], label [[LAND_LHS_TRUE10:%.*]], label [[LAND_END:%.*]]
; CHECK:   "land.lhs.true+land.lhs.true10+land.lhs.true4":
; CHECK-NEXT:    [[MEMCMP1:%.*]] = call i32 @memcmp(ptr [[A]], ptr [[B:%.*]], i64 6)
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i32 [[MEMCMP1]], 0
; CHECK-NEXT:    br label [[LAND_END]]
; CHECK:       land.end:
; CHECK-NEXT:    [[RES:%.*]] = phi i1 [ [[CMP1]], [[LAND_LHS_TRUE10]] ], [ false, [[ENTRY_LAND_RHS:%.*]] ]
; CHECK-NEXT:    ret i1 [[RES]]
;
entry:
  %e = getelementptr inbounds nuw i8, ptr %a, i64 8
  %0 = load i32, ptr %e, align 4
  %cmp = icmp eq i32 %0, 255
  call void (...) @foo() inaccessiblememonly
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
  br i1 %cmp7, label %land.lhs.true8, label %land.end

land.lhs.true8:                                   ; preds = %land.lhs.true4
  %g = getelementptr inbounds nuw i8, ptr %a, i64 16
  %5 = load i32, ptr %g, align 4
  %cmp9 = icmp eq i32 %5, 100
  br i1 %cmp9, label %land.lhs.true10, label %land.end

land.lhs.true10:                                  ; preds = %land.lhs.true8
  %b11 = getelementptr inbounds nuw i8, ptr %a, i64 4
  %6 = load i8, ptr %b11, align 4
  %b13 = getelementptr inbounds nuw i8, ptr %b, i64 4
  %7 = load i8, ptr %b13, align 4
  %cmp15 = icmp eq i8 %6, %7
  br i1 %cmp15, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %land.lhs.true10
  %f = getelementptr inbounds nuw i8, ptr %a, i64 12
  %8 = load i32, ptr %f, align 4
  %cmp16 = icmp eq i32 %8, 200
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true10, %land.lhs.true8, %land.lhs.true4, %land.lhs.true, %entry
  %9 = phi i1 [ false, %land.lhs.true10 ], [ false, %land.lhs.true8 ], [ false, %land.lhs.true4 ], [ false, %land.lhs.true ], [ false, %entry ], [ %cmp16, %land.rhs ]
  ret i1 %9
}

; If block to split it in BCE-comparison that that block should be first.

define dso_local noundef zeroext i1 @cmp_mixed_bce_first(
    ptr noundef nonnull readonly align 4 captures(none) dereferenceable(20) %a,
    ptr noundef nonnull readonly align 4 captures(none) dereferenceable(20) %b) local_unnamed_addr {
; CHECK-LABEL: @cmp_mixed_bce_first(
; CHECK:   "entry+land.lhs.true10+land.lhs.true4":
; CHECK-NEXT:    call void (...) @foo() #[[ATTR2:[0-9]+]]
; CHECK-NEXT:    [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[A:%.*]], ptr [[B:%.*]], i64 6)
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i32 [[MEMCMP]], 0
; CHECK-NEXT:    br i1 [[CMP1]], label [[LAND_LHS_TRUE:%.*]], label [[LAND_END:%.*]]
; CHECK:  "land.lhs.true+land.rhs+land.lhs.true4":
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 8
; CHECK-NEXT:    [[TMP1:%.*]] = alloca <{ i32, i32, i32 }>
; CHECK-NEXT:    store <{ i32, i32, i32 }> <{ i32 255, i32 200, i32 100 }>, ptr [[TMP1]], align 1
; CHECK-NEXT:    [[MEMCMP2:%.*]] = call i32 @memcmp(ptr [[TMP0]], ptr [[TMP1]], i64 12)
; CHECK-NEXT:    [[CMP2:%.*]] = icmp eq i32 [[MEMCMP2]], 0
; CHECK-NEXT:    br label [[LAND_END]]
; CHECK:       land.end:
; CHECK-NEXT:    [[TMP4:%.*]] = phi i1 [ [[CMP2]], [[LAND_LHS_TRUE]] ], [ false, [[ENTRY:%.*]] ]
; CHECK-NEXT:    ret i1 [[TMP4]]
;
entry:
  %0 = load i32, ptr %a, align 4
  %1 = load i32, ptr %b, align 4
  call void (...) @foo() inaccessiblememonly
  %cmp3 = icmp eq i32 %0, %1
  br i1 %cmp3, label %land.lhs.true, label %land.end

land.lhs.true:
  %e = getelementptr inbounds nuw i8, ptr %a, i64 8
  %2 = load i32, ptr %e, align 4
  %cmp = icmp eq i32 %2, 255
  br i1 %cmp, label %land.lhs.true4, label %land.end

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
  %b11 = getelementptr inbounds nuw i8, ptr %a, i64 4
  %6 = load i8, ptr %b11, align 4
  %b13 = getelementptr inbounds nuw i8, ptr %b, i64 4
  %7 = load i8, ptr %b13, align 4
  %cmp15 = icmp eq i8 %6, %7
  br i1 %cmp15, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %land.lhs.true10
  %f = getelementptr inbounds nuw i8, ptr %a, i64 12
  %8 = load i32, ptr %f, align 4
  %cmp16 = icmp eq i32 %8, 200
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true10, %land.lhs.true4, %land.lhs.true, %entry
  %9 = phi i1 [ false, %land.lhs.true10 ], [ false, %land.lhs.true4 ], [ false, %land.lhs.true ], [ false, %entry ], [ %cmp16, %land.rhs ]
  ret i1 %9
}
