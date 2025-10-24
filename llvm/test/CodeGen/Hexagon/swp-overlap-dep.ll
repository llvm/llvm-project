; Test passes if the inner loop consists of 5 packets.
; This is due to the fact that the pipeliner is able
; to create a schedule with II=5

; RUN: llc -O3 -mtriple=hexagon -mv71t < %s | FileCheck %s

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: {
; CHECK: }
; CHECK: {
; CHECK: }
; CHECK: {
; CHECK: }
; CHECK: {
; CHECK: }
; CHECK: {
; CHECK: }{{[ \t]*}}:endloop0

define dso_local void @foo(ptr noundef readonly captures(none) %arg, ptr noundef readonly captures(none) %arg1, ptr noundef captures(none) %arg2, i32 noundef %arg3, i32 noundef %arg4, ptr noalias noundef captures(none) %arg5) local_unnamed_addr {
bb:
  %icmp = icmp sgt i32 %arg3, 0
  br i1 %icmp, label %bb6, label %bb70

bb6:                                              ; preds = %bb
  %icmp7 = icmp sgt i32 %arg4, 0
  br i1 %icmp7, label %bb8, label %bb57

bb8:                                              ; preds = %bb53, %bb6
  %phi = phi ptr [ %getelementptr56, %bb53 ], [ %arg2, %bb6 ]
  %phi9 = phi ptr [ %getelementptr32, %bb53 ], [ %arg1, %bb6 ]
  %phi10 = phi i32 [ %add54, %bb53 ], [ 0, %bb6 ]
  %phi11 = phi ptr [ %arg5, %bb53 ], [ %arg, %bb6 ]
  %load = load i32, ptr %phi, align 4
  %getelementptr = getelementptr inbounds nuw i8, ptr %phi, i32 4
  %load12 = load i32, ptr %getelementptr, align 4
  %zext = zext i32 %load12 to i64
  %shl = shl nuw i64 %zext, 32
  %zext13 = zext i32 %load to i64
  %or = or disjoint i64 %shl, %zext13
  %getelementptr14 = getelementptr inbounds nuw i8, ptr %phi9, i32 4
  %load15 = load i32, ptr %getelementptr14, align 4
  %getelementptr16 = getelementptr inbounds nuw i8, ptr %phi9, i32 8
  %load17 = load i32, ptr %getelementptr16, align 4
  %sub = sub nsw i32 0, %load15
  %zext18 = zext i32 %load17 to i64
  %shl19 = shl nuw i64 %zext18, 32
  %zext20 = zext i32 %sub to i64
  %or21 = or disjoint i64 %shl19, %zext20
  %getelementptr22 = getelementptr inbounds nuw i8, ptr %phi9, i32 12
  %load23 = load i32, ptr %getelementptr22, align 4
  %getelementptr24 = getelementptr inbounds nuw i8, ptr %phi9, i32 16
  %load25 = load i32, ptr %getelementptr24, align 4
  %getelementptr26 = getelementptr inbounds nuw i8, ptr %phi9, i32 20
  %load27 = load i32, ptr %getelementptr26, align 4
  %zext28 = zext i32 %load27 to i64
  %shl29 = shl nuw i64 %zext28, 32
  %zext30 = zext i32 %load25 to i64
  %or31 = or disjoint i64 %shl29, %zext30
  %getelementptr32 = getelementptr i8, ptr %phi9, i32 24
  br label %bb33

bb33:                                             ; preds = %bb33, %bb8
  %phi34 = phi ptr [ %arg5, %bb8 ], [ %getelementptr52, %bb33 ]
  %phi35 = phi i32 [ 0, %bb8 ], [ %add, %bb33 ]
  %phi36 = phi i32 [ %load, %bb8 ], [ %call42, %bb33 ]
  %phi37 = phi i64 [ %or, %bb8 ], [ %or50, %bb33 ]
  %phi38 = phi ptr [ %phi11, %bb8 ], [ %getelementptr39, %bb33 ]
  %getelementptr39 = getelementptr inbounds nuw i8, ptr %phi38, i32 4
  %load40 = load i32, ptr %phi38, align 4
  %sext = sext i32 %load40 to i64
  %shl41 = shl nsw i64 %sext, 25
  %call = tail call i64 @llvm.hexagon.M7.dcmpyrw.acc(i64 %shl41, i64 %or21, i64 %phi37)
  %ashr = ashr i64 %call, 28
  %call42 = tail call i32 @llvm.hexagon.A2.sat(i64 %ashr)
  %call43 = tail call i64 @llvm.hexagon.M7.dcmpyrwc(i64 %or31, i64 %phi37)
  %call44 = tail call i64 @llvm.hexagon.M2.dpmpyss.acc.s0(i64 %call43, i32 %load23, i32 %call42)
  %ashr45 = ashr i64 %call44, 25
  %call46 = tail call i32 @llvm.hexagon.A2.sat(i64 %ashr45)
  store i32 %call46, ptr %phi34, align 4
  %zext47 = zext i32 %phi36 to i64
  %shl48 = shl nuw i64 %zext47, 32
  %zext49 = zext i32 %call42 to i64
  %or50 = or disjoint i64 %shl48, %zext49
  %add = add nuw nsw i32 %phi35, 1
  %icmp51 = icmp eq i32 %add, %arg4
  %getelementptr52 = getelementptr i8, ptr %phi34, i32 4
  br i1 %icmp51, label %bb53, label %bb33, !llvm.loop !1

bb53:                                             ; preds = %bb33
  store i64 %or50, ptr %phi, align 8
  %add54 = add nuw nsw i32 %phi10, 1
  %icmp55 = icmp eq i32 %add54, %arg3
  %getelementptr56 = getelementptr i8, ptr %phi, i32 8
  br i1 %icmp55, label %bb70, label %bb8, !llvm.loop !3

bb57:                                             ; preds = %bb57, %bb6
  %phi58 = phi ptr [ %getelementptr69, %bb57 ], [ %arg2, %bb6 ]
  %phi59 = phi i32 [ %add67, %bb57 ], [ 0, %bb6 ]
  %load60 = load i32, ptr %phi58, align 4
  %getelementptr61 = getelementptr inbounds nuw i8, ptr %phi58, i32 4
  %load62 = load i32, ptr %getelementptr61, align 4
  %zext63 = zext i32 %load62 to i64
  %shl64 = shl nuw i64 %zext63, 32
  %zext65 = zext i32 %load60 to i64
  %or66 = or disjoint i64 %shl64, %zext65
  store i64 %or66, ptr %phi58, align 8
  %add67 = add nuw nsw i32 %phi59, 1
  %icmp68 = icmp eq i32 %add67, %arg3
  %getelementptr69 = getelementptr i8, ptr %phi58, i32 8
  br i1 %icmp68, label %bb70, label %bb57, !llvm.loop !4

bb70:                                             ; preds = %bb57, %bb53, %bb
  ret void
}

declare i64 @llvm.hexagon.M7.dcmpyrw.acc(i64, i64, i64)
declare i32 @llvm.hexagon.A2.sat(i64)
declare i64 @llvm.hexagon.M7.dcmpyrwc(i64, i64)
declare i64 @llvm.hexagon.M2.dpmpyss.acc.s0(i64, i32, i32)

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.mustprogress"}
!3 = distinct !{!3, !2}
!4 = distinct !{!4, !2}
