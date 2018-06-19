; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; CHECK: %tmp9 = shl i64 %tmp8, 33
; CHECK-NEXT: --> {{.*}} Exits: (-8589934592 + (8589934592 * (zext i32 %arg2 to i64)))
; CHECK: %tmp10 = ashr exact i64 %tmp9, 32
; CHECK-NEXT: --> {{.*}} Exits: (sext i32 (-2 + (2 * %arg2)) to i64)
; CHECK: %tmp11 = getelementptr inbounds i32, i32* %arg, i64 %tmp10
; CHECK-NEXT: --> {{.*}} Exits: ((4 * (sext i32 (-2 + (2 * %arg2)) to i64)) + %arg)
; CHECK:  %tmp14 = or i64 %tmp10, 1
; CHECK-NEXT: --> {{.*}} Exits: (1 + (sext i32 (-2 + (2 * %arg2)) to i64))<nsw>
; CHECK: %tmp15 = getelementptr inbounds i32, i32* %arg, i64 %tmp14
; CHECK-NEXT: --> {{.*}} Exits: (4 + (4 * (sext i32 (-2 + (2 * %arg2)) to i64)) + %arg)
; CHECK:Loop %bb7: backedge-taken count is (-1 + (zext i32 %arg2 to i64))<nsw>
; CHECK-NEXT:Loop %bb7: max backedge-taken count is -1
; CHECK-NEXT:Loop %bb7: Predicated backedge-taken count is (-1 + (zext i32 %arg2 to i64))<nsw>

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
  %tmp10 = ashr exact i64 %tmp9, 32
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

; CHECK: %t10 = ashr exact i128 %t9, 1
; CHECK-NEXT: --> {{.*}} Exits: (sext i127 (-633825300114114700748351602688 + (633825300114114700748351602688 * (zext i32 %arg5 to i127))) to i128)
; CHECK: %t14 = or i128 %t10, 1
; CHECK-NEXT: --> {{.*}} Exits: (1 + (sext i127 (-633825300114114700748351602688 + (633825300114114700748351602688 * (zext i32 %arg5 to i127))) to i128))<nsw>
; CHECK: Loop %bb7: backedge-taken count is (-1 + (zext i32 %arg5 to i128))<nsw>
; CHECK-NEXT: Loop %bb7: max backedge-taken count is -1
; CHECK-NEXT: Loop %bb7: Predicated backedge-taken count is (-1 + (zext i32 %arg5 to i128))<nsw>

define void @goo(i32* nocapture %arg3, i32 %arg4, i32 %arg5) {
bb:
  %t = icmp sgt i32 %arg5, 0
  br i1 %t, label %bb3, label %bb6

bb3:                                              ; preds = %bb
  %t4 = zext i32 %arg5 to i128
  br label %bb7

bb5:                                              ; preds = %bb7
  br label %bb6

bb6:                                              ; preds = %bb5, %bb
  ret void

bb7:                                              ; preds = %bb7, %bb3
  %t8 = phi i128 [ %t18, %bb7 ], [ 0, %bb3 ]
  %t9 = shl i128 %t8, 100
  %t10 = ashr exact i128 %t9, 1
  %t11 = getelementptr inbounds i32, i32* %arg3, i128 %t10
  %t12 = load i32, i32* %t11, align 4
  %t13 = sub nsw i32 %t12, %arg4
  store i32 %t13, i32* %t11, align 4
  %t14 = or i128 %t10, 1
  %t15 = getelementptr inbounds i32, i32* %arg3, i128 %t14
  %t16 = load i32, i32* %t15, align 4
  %t17 = mul nsw i32 %t16, %arg4
  store i32 %t17, i32* %t15, align 4
  %t18 = add nuw nsw i128 %t8, 1
  %t19 = icmp eq i128 %t18, %t4
  br i1 %t19, label %bb5, label %bb7
}
