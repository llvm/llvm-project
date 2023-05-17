; REQUIRES: asserts
; RUN: opt -debug-only=loop-unroll-and-jam -passes="loop-unroll-and-jam" -enable-unroll-and-jam -allow-unroll-and-jam -unroll-and-jam-count=8 -disable-output %s 2>&1 | FileCheck %s

; CHECK: Loop Unroll and Jam: F[h] Loop %bb8
; CHECK: Won't unroll-and-jam; only loops with single exit blocks can be unrolled and jammed

@a = external global i16, align 2
@e = external global i32, align 4
@f = external global i16, align 2
@b = external global i16, align 2
@c = external global i64, align 8

define void @h() {
bb:
  store i32 4, ptr @e, align 4
  %i15 = load i16, ptr @b, align 2
  %i17 = icmp slt i16 %i15, 1
  br label %bb8

bb8:                                              ; preds = %bb, %bb47
  %storemerge15 = phi i32 [ 4, %bb ], [ %i49, %bb47 ]
  br label %bb24

bb24:                                             ; preds = %bb43, %bb8
  %storemerge312 = phi i16 [ 0, %bb8 ], [ %i45, %bb43 ]
  br i1 %i17, label %bb46.preheader, label %bb43

bb46.preheader:                                   ; preds = %bb24
  store i16 %storemerge312, ptr @f, align 2
  br label %bb46

bb43:                                             ; preds = %bb24
  %i45 = add nuw nsw i16 %storemerge312, 1
  %i13 = icmp ult i16 %storemerge312, 7
  br i1 %i13, label %bb24, label %bb47

bb46:                                             ; preds = %bb46.preheader, %bb46
  br label %bb46

bb47:                                             ; preds = %bb43
  %i49 = add nsw i32 %storemerge15, -1
  store i32 %i49, ptr @e, align 4
  %i7.not = icmp eq i32 %i49, 0
  br i1 %i7.not, label %bb50, label %bb8

bb50:                                             ; preds = %bb47
  store i16 %i45, ptr @f, align 2
  ret void
}
