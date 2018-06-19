; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; CHECK:  %tmp9 = shl i64 %tmp8, 33
; CHECK-NEXT:  -->  {{.*}} Exits: (-8589934592 + (8589934592 * (zext i32 %arg2 to i64)))
; CHECK-NEXT:  %tmp10 = ashr exact i64 %tmp9, 0
; CHECK-NEXT:  -->  {{.*}} Exits: (-8589934592 + (8589934592 * (zext i32 %arg2 to i64)))

define void @foo(i32* nocapture %arg, i32 %arg1, i32 %arg2) {
bb:
  %tmp = icmp sgt i32 %arg2, 0
  br i1 %tmp, label %bb3, label %bb6

bb3:                                              ; preds = %bb
  %tmp4 = zext i32 %arg2 to i64
  br label %bb7

bb5:                                              ; preds = %bb7
  br label %bb6

bb6:                                              ; preds = %bb5, %bb
  ret void

bb7:                                              ; preds = %bb7, %bb3
  %tmp8 = phi i64 [ %tmp18, %bb7 ], [ 0, %bb3 ]
  %tmp9 = shl i64 %tmp8, 33
  %tmp10 = ashr exact i64 %tmp9, 0
  %tmp11 = getelementptr inbounds i32, i32* %arg, i64 %tmp10
  %tmp12 = load i32, i32* %tmp11, align 4
  %tmp13 = sub nsw i32 %tmp12, %arg1
  store i32 %tmp13, i32* %tmp11, align 4
  %tmp14 = or i64 %tmp10, 1
  %tmp15 = getelementptr inbounds i32, i32* %arg, i64 %tmp14
  %tmp16 = load i32, i32* %tmp15, align 4
  %tmp17 = mul nsw i32 %tmp16, %arg1
  store i32 %tmp17, i32* %tmp15, align 4
  %tmp18 = add nuw nsw i64 %tmp8, 1
  %tmp19 = icmp eq i64 %tmp18, %tmp4
  br i1 %tmp19, label %bb5, label %bb7
}
