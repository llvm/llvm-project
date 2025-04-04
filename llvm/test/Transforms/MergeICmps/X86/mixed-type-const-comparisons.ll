; RUN: opt < %s -mtriple=x86_64-unknown-unknown -passes=mergeicmps -verify-dom-info -S | FileCheck %s

; Tests if a const-cmp-chain of different types can still be merged.
; This is usually the case when comparing different struct fields to constants.

; CHECK: [[MEMCMP_OP0:@memcmp_const_op]] = private constant <{ i32, i8 }> <{ i32 3, i8 100 }>
; CHECK: [[MEMCMP_OP1:@memcmp_const_op.1]] = private constant <{ i32, i8, i8 }> <{ i32 200, i8 3, i8 100 }>

; Can only merge gep 0 with gep 4 due to alignment since gep 8 is not directly adjacent to gep 4.
define dso_local zeroext i1 @is_all_ones_struct(
; CHECK-LABEL: @is_all_ones_struct(
; CHECK:      entry1:
; CHECK-NEXT:   [[TMP0:%.*]] = getelementptr inbounds nuw i8, ptr [[P:%.*]], i64 8
; CHECK-NEXT:   [[TMP1:%.*]] = load i32, ptr [[TMP0]], align 4
; CHECK-NEXT:   [[CMP0:%.*]] = icmp eq i32 [[TMP1]], 200
; CHECK-NEXT:   br i1 [[CMP0]], label [[MERGED:%.*]], label [[LAND_END:%.*]]
; CHECK:      "land.rhs+land.lhs.true":
; CHECK-NEXT:   [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[P]], ptr [[MEMCMP_OP0]], i64 5)
; CHECK-NEXT:   [[CMP1:%.*]] = icmp eq i32 [[MEMCMP]], 0
; CHECK-NEXT:   br label [[LAND_END]]
; CHECK:      land.end:
; CHECK-NEXT:   [[RES:%.*]] = phi i1 [ [[CMP1]], [[MERGED]] ], [ false, %entry1 ]
; CHECK-NEXT:   ret i1 [[RES]]
;
  ptr noundef nonnull readonly align 4 captures(none) dereferenceable(24) %p) local_unnamed_addr {
entry:
  %c = getelementptr inbounds nuw i8, ptr %p, i64 8
  %0 = load i32, ptr %c, align 4
  %cmp = icmp eq i32 %0, 200
  br i1 %cmp, label %land.lhs.true, label %land.end

land.lhs.true:                                    ; preds = %entry
  %b = getelementptr inbounds nuw i8, ptr %p, i64 4
  %1 = load i8, ptr %b, align 4
  %cmp1 = icmp eq i8 %1, 100
  br i1 %cmp1, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %land.lhs.true
  %2 = load i32, ptr %p, align 4
  %cmp3 = icmp eq i32 %2, 3
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true, %entry
  %3 = phi i1 [ false, %land.lhs.true ], [ false, %entry ], [ %cmp3, %land.rhs ]
  ret i1 %3
}


; Can also still merge select blocks with different types.
define dso_local noundef zeroext i1 @is_all_ones_struct_select_block(
; CHECK-LABEL: @is_all_ones_struct_select_block(
; CHECK:      "entry+land.rhs":
; CHECK-NEXT:   [[MEMCMP:%.*]] = call i32 @memcmp(ptr [[P:%.*]], ptr [[MEMCMP_OP1]], i64 6)
; CHECK-NEXT:   [[CMP1:%.*]] = icmp eq i32 [[MEMCMP]], 0
; CHECK-NEXT:   br label [[LAND_END]]
; CHECK:      land.end:
; CHECK-NEXT:   ret i1 [[CMP1]]
;
  ptr noundef nonnull readonly align 4 captures(none) dereferenceable(24) %p) local_unnamed_addr {
entry:
  %0 = load i32, ptr %p, align 4
  %cmp = icmp eq i32 %0, 200
  %c = getelementptr inbounds nuw i8, ptr %p, i64 5
  %1 = load i8, ptr %c, align 1
  %cmp2 = icmp eq i8 %1, 100
  %or.cond = select i1 %cmp, i1 %cmp2, i1 false
  br i1 %or.cond, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %entry
  %b3 = getelementptr inbounds nuw i8, ptr %p, i64 4
  %2 = load i8, ptr %b3, align 4
  %cmp5 = icmp eq i8 %2, 3
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %3 = phi i1 [ false, %entry ], [ %cmp5, %land.rhs ]
  ret i1 %3
}
