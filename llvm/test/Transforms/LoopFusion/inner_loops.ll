; RUN: opt -S -passes=loop-fusion < %s 2>&1 | FileCheck %s

@A = common global [1024 x [1024 x i32]] zeroinitializer, align 16
@B = common global [1024 x [1024 x i32]] zeroinitializer, align 16

; CHECK: void @dep_free
; CHECK-NEXT: bb:
; CHECK-NEXT: br label %[[LOOP1HEADER:bb[0-9]*]]
; CHECK: [[LOOP1HEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1BODY:bb[0-9]*]], label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP1BODY]]
; CHECK: br label %[[LOOP1LATCH:bb[0-9]*]]
; CHECK: [[LOOP1LATCH]]
; CHECK: br label %[[LOOP2PREHEADER:bb[0-9]+]]
; CHECK: [[LOOP2PREHEADER]]
; CHECK: br i1 %{{.*}}, label %[[LOOP2BODY:bb[0-9]*]], label %[[LOOP2EXIT:bb[0-9]*]]
; CHECK: [[LOOP2BODY]]
; CHECK: br label %[[LOOP2LATCH:bb[0-9]+]]
; CHECK: [[LOOP2LATCH]]
; CHECK: br label %[[LOOP1HEADER]]
; CHECK: ret void

define void @dep_free() {
bb:
  br label %bb9

bb9:                                              ; preds = %bb35, %bb
  %indvars.iv6 = phi i64 [ %indvars.iv.next7, %bb35 ], [ 0, %bb ]
  %.0 = phi i32 [ 0, %bb ], [ %tmp36, %bb35 ]
  %exitcond8 = icmp ne i64 %indvars.iv6, 100
  br i1 %exitcond8, label %bb11, label %bb10

bb10:                                             ; preds = %bb9
  br label %bb37

bb11:                                             ; preds = %bb9
  br label %bb12

bb12:                                             ; preds = %bb21, %bb11
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb21 ], [ 0, %bb11 ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %bb14, label %bb23

bb14:                                             ; preds = %bb12
  %tmp = add nsw i32 %.0, -3
  %tmp15 = add nuw nsw i64 %indvars.iv6, 3
  %tmp16 = trunc i64 %tmp15 to i32
  %tmp17 = mul nsw i32 %tmp, %tmp16
  %tmp18 = trunc i64 %indvars.iv6 to i32
  %tmp19 = srem i32 %tmp17, %tmp18
  %tmp20 = getelementptr inbounds [1024 x [1024 x i32]], ptr @A, i64 0, i64 %indvars.iv6, i64 %indvars.iv
  store i32 %tmp19, ptr %tmp20, align 4
  br label %bb21

bb21:                                             ; preds = %bb14
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb12

bb23:                                             ; preds = %bb33, %bb12
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %bb33 ], [ 0, %bb12 ]
  %exitcond5 = icmp ne i64 %indvars.iv3, 100
  br i1 %exitcond5, label %bb25, label %bb35

bb25:                                             ; preds = %bb23
  %tmp26 = add nsw i32 %.0, -3
  %tmp27 = add nuw nsw i64 %indvars.iv6, 3
  %tmp28 = trunc i64 %tmp27 to i32
  %tmp29 = mul nsw i32 %tmp26, %tmp28
  %tmp30 = trunc i64 %indvars.iv6 to i32
  %tmp31 = srem i32 %tmp29, %tmp30
  %tmp32 = getelementptr inbounds [1024 x [1024 x i32]], ptr @B, i64 0, i64 %indvars.iv6, i64 %indvars.iv3
  store i32 %tmp31, ptr %tmp32, align 4
  br label %bb33

bb33:                                             ; preds = %bb25
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  br label %bb23

bb35:                                             ; preds = %bb23
  %indvars.iv.next7 = add nuw nsw i64 %indvars.iv6, 1
  %tmp36 = add nuw nsw i32 %.0, 1
  br label %bb9

bb37:                                             ; preds = %bb10
  ret void
}
