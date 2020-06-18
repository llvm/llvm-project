; RUN: opt < %s -canon-freeze -S | FileCheck %s
; REQUIRES: aarch64-registered-target
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

%struct.arc = type { i32 }
%struct.g = type { i64, %struct.arc, i64, i64, i64 }

@m = global i64 0
@h = global %struct.arc* null
@j = global %struct.g zeroinitializer

define dso_local i32 @main() {
bb:
  %tmp = load i64, i64* getelementptr inbounds (%struct.g, %struct.g* @j, i32 0, i32 0), align 8
  %tmp1 = icmp sgt i64 %tmp, 0
  br i1 %tmp1, label %bb2, label %bb35

bb2:                                              ; preds = %bb
  %tmp3 = load i64, i64* @m, align 8
  %tmp4 = load %struct.arc*, %struct.arc** @h, align 8
; CHECK: %tmp3.frozen = freeze i64 %tmp3
  br label %bb5

bb5:                                              ; preds = %bb28, %bb2
  %tmp6 = phi %struct.arc* [ %tmp4, %bb2 ], [ %tmp31, %bb28 ]
  %tmp7 = phi i64 [ %tmp3, %bb2 ], [ %tmp12, %bb28 ]
; CHECK: %tmp7 = phi i64 [ %tmp3.frozen, %bb2 ], [ %tmp12, %bb28 ]
  %tmp8 = phi i64 [ 0, %bb2 ], [ %tmp11, %bb28 ]
  %tmp9 = trunc i64 %tmp7 to i32
  %tmp10 = getelementptr inbounds %struct.arc, %struct.arc* %tmp6, i64 0, i32 0
  store i32 %tmp9, i32* %tmp10, align 4
  %tmp11 = add nuw nsw i64 %tmp8, 1
  %tmp12 = add nsw i64 %tmp7, 1
; CHECK: %tmp12 = add i64 %tmp7, 1
  store i64 %tmp12, i64* @m, align 8
  %tmp13 = load i64, i64* inttoptr (i64 16 to i64*), align 16
  %tmp14 = freeze i64 %tmp12
; CHECK-NOT: %tmp14 = freeze i64 %tmp12
  %tmp15 = freeze i64 %tmp13
  %tmp16 = sdiv i64 %tmp14, %tmp15
  %tmp17 = mul i64 %tmp16, %tmp15
  %tmp18 = sub i64 %tmp14, %tmp17
  %tmp19 = load i64, i64* inttoptr (i64 24 to i64*), align 8
  %tmp20 = icmp sgt i64 %tmp18, %tmp19
  %tmp21 = load i64, i64* inttoptr (i64 32 to i64*), align 32
  br i1 %tmp20, label %bb22, label %bb28

bb22:                                             ; preds = %bb5
  %tmp23 = mul nsw i64 %tmp21, %tmp19
  %tmp24 = sub nsw i64 %tmp18, %tmp19
  %tmp25 = add nsw i64 %tmp21, -1
  %tmp26 = mul nsw i64 %tmp25, %tmp24
  %tmp27 = add nsw i64 %tmp26, %tmp23
  br label %bb28

bb28:                                             ; preds = %bb22, %bb5
  %tmp29 = phi i64 [ %tmp27, %bb22 ], [ %tmp21, %bb5 ]
  %tmp30 = add nsw i64 %tmp29, %tmp16
  %tmp31 = getelementptr inbounds %struct.arc, %struct.arc* getelementptr inbounds (%struct.g, %struct.g* @j, i32 0, i32 1), i64 %tmp30
  store %struct.arc* %tmp31, %struct.arc** @h, align 8
  %tmp32 = load i64, i64* getelementptr inbounds (%struct.g, %struct.g* @j, i32 0, i32 0), align 8
  %tmp33 = icmp slt i64 %tmp11, %tmp32
  br i1 %tmp33, label %bb5, label %bb34

bb34:                                             ; preds = %bb28
  br label %bb35

bb35:                                             ; preds = %bb34, %bb
  ret i32 0
}
