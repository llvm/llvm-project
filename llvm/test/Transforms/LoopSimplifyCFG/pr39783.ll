; REQUIRES: asserts
; RUN: opt -mcpu=z13 -S -loop-simplifycfg -enable-loop-simplifycfg-term-folding -verify-memoryssa 2>&1 < %s | FileCheck %s
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"

@global = external dso_local local_unnamed_addr global i8, align 2
@global.1 = external dso_local local_unnamed_addr global i32, align 4
@global.2 = external dso_local local_unnamed_addr global i32, align 4
@global.3 = external dso_local local_unnamed_addr global i16, align 2
@global.4 = external dso_local local_unnamed_addr global i32, align 4

; CHECK-LABEL: @test_01(

define internal fastcc void @test_01() unnamed_addr {
bb:
  %tmp = load i32, i32* @global.2, align 4
  %tmp1 = icmp eq i32 %tmp, 0
  br i1 %tmp1, label %bb3, label %bb2

bb2:                                              ; preds = %bb
  br label %bb7

bb3:                                              ; preds = %bb
  br label %bb4

bb4:                                              ; preds = %bb6, %bb3
  br i1 true, label %bb5, label %bb6

bb5:                                              ; preds = %bb4
  store i16 0, i16* @global.3, align 2
  br label %bb6

bb6:                                              ; preds = %bb5, %bb4
  br label %bb4

bb7:                                              ; preds = %bb7, %bb2
  %tmp8 = phi i32 [ 1, %bb7 ], [ 0, %bb2 ]
  %tmp9 = icmp eq i32 %tmp8, 0
  br i1 %tmp9, label %bb7, label %bb10

bb10:                                             ; preds = %bb7
  br label %bb11

bb11:                                             ; preds = %bb13, %bb10
  %tmp12 = icmp ult i32 %tmp, 6
  br i1 %tmp12, label %bb13, label %bb14

bb13:                                             ; preds = %bb11
  store i32 0, i32* @global.1, align 4
  br label %bb11

bb14:                                             ; preds = %bb11
  ret void
}

@global.5 = external dso_local local_unnamed_addr global i16, align 2

declare dso_local void @spam() local_unnamed_addr

declare dso_local void @blam() local_unnamed_addr

declare dso_local i64 @quux.1() local_unnamed_addr

declare dso_local void @bar() local_unnamed_addr

; CHECK-LABEL: @test_02(

define dso_local void @test_02(i8 signext %arg) local_unnamed_addr {
bb:
  br label %bb1

bb1:                                              ; preds = %bb16, %bb
  %tmp = phi i8 [ %arg, %bb ], [ %tmp17, %bb16 ]
  %tmp2 = load i16, i16* @global.5, align 2
  %tmp3 = icmp ugt i16 %tmp2, 56
  br i1 %tmp3, label %bb4, label %bb18

bb4:                                              ; preds = %bb1
  %tmp5 = tail call i64 @quux.1()
  %tmp6 = icmp eq i64 %tmp5, 0
  br i1 %tmp6, label %bb13, label %bb7

bb7:                                              ; preds = %bb4
  br label %bb8

bb8:                                              ; preds = %bb8, %bb7
  %tmp9 = phi i32 [ 26, %bb7 ], [ %tmp10, %bb8 ]
  tail call void @bar()
  %tmp10 = add nsw i32 %tmp9, -1
  %tmp11 = icmp eq i32 %tmp10, 12
  br i1 %tmp11, label %bb12, label %bb8

bb12:                                             ; preds = %bb8
  br i1 false, label %bb14, label %bb16

bb13:                                             ; preds = %bb4
  tail call void @spam()
  br label %bb14

bb14:                                             ; preds = %bb13, %bb12
  %tmp15 = phi i8 [ -23, %bb12 ], [ %tmp, %bb13 ]
  tail call void @blam()
  br label %bb16

bb16:                                             ; preds = %bb14, %bb12
  %tmp17 = phi i8 [ %tmp15, %bb14 ], [ -23, %bb12 ]
  br label %bb1

bb18:                                             ; preds = %bb1
  ret void
}
